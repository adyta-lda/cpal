//! Monitors AVAudioSession lifecycle events and reports them as stream errors.

use std::{
    ptr::NonNull,
    sync::{Arc, Mutex},
};

use block2::RcBlock;
use objc2::runtime::AnyObject;
use objc2_avf_audio::{
    AVAudioSessionMediaServicesWereLostNotification,
    AVAudioSessionMediaServicesWereResetNotification, AVAudioSessionRouteChangeNotification,
    AVAudioSessionRouteChangeReason, AVAudioSessionRouteChangeReasonKey,
};
use objc2_foundation::{NSNotification, NSNotificationCenter, NSNumber, NSString};

use crate::{Error, ErrorKind};

pub(super) type ErrorCallbackMutex = Arc<Mutex<Box<dyn FnMut(Error) + Send>>>;

/// Which AVAudioSession route changes should invalidate the stream.
///
/// Only meaningful when something other than cpal owns the session: an
/// application that configures its own routing causes route changes
/// deliberately, and does not want cpal tearing the stream down underneath it
/// each time it does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RouteChangePolicy {
    /// Report every route change cpal recognises. Correct when cpal owns the
    /// session, since then any change came from outside.
    AnyChange,
    /// Report only changes the owning application cannot have caused itself.
    ExternalOnly,
}

unsafe fn route_change_error(
    notification: &NSNotification,
    policy: RouteChangePolicy,
) -> Option<Error> {
    let user_info = notification.userInfo()?;
    let key = AVAudioSessionRouteChangeReasonKey?;
    let dict = unsafe { user_info.cast_unchecked::<NSString, AnyObject>() };
    let value = dict.objectForKey(key)?;
    let number = value.downcast_ref::<NSNumber>()?;
    let reason = AVAudioSessionRouteChangeReason(number.unsignedIntegerValue());
    match reason {
        AVAudioSessionRouteChangeReason::OldDeviceUnavailable => Some(Error::with_message(
            ErrorKind::DeviceChanged,
            "audio route changed",
        )),

        // These three are exactly the reasons an application that owns the
        // session produces itself: `.CategoryChange` from `setCategory`,
        // `.Override` from `overrideOutputAudioPort`, `.RouteConfigurationChange`
        // from `setPreferredInput`. Reporting them under `ExternalOnly` would
        // make cpal rebuild in reaction to the owner's own deliberate calls —
        // concurrently with them, since the notification arrives while the
        // owner may still be part-way through a multi-step route change. The
        // owner is expected to tell cpal when it has finished instead.
        AVAudioSessionRouteChangeReason::CategoryChange
        | AVAudioSessionRouteChangeReason::Override
        | AVAudioSessionRouteChangeReason::RouteConfigurationChange => match policy {
            RouteChangePolicy::AnyChange => Some(Error::with_message(
                ErrorKind::StreamInvalidated,
                "audio route changed",
            )),
            RouteChangePolicy::ExternalOnly => None,
        },

        AVAudioSessionRouteChangeReason::NoSuitableRouteForCategory => Some(Error::with_message(
            ErrorKind::DeviceNotAvailable,
            "no suitable audio route for the session category",
        )),

        _ => None,
    }
}

pub(super) struct SessionEventManager {
    observers: Vec<
        objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2::runtime::NSObjectProtocol>>,
    >,
}

// SAFETY: NSNotificationCenter is thread-safe on iOS. The observer tokens stored here are opaque
// handles used only to call removeObserver in Drop; no data is read or written through them.
unsafe impl Send for SessionEventManager {}
unsafe impl Sync for SessionEventManager {}

impl SessionEventManager {
    pub(super) fn new(error_callback: ErrorCallbackMutex, policy: RouteChangePolicy) -> Self {
        let nc = NSNotificationCenter::defaultCenter();
        let mut observers = Vec::new();

        {
            let cb = error_callback.clone();
            let block = RcBlock::new(move |notif: NonNull<NSNotification>| {
                if let Some(err) = unsafe { route_change_error(notif.as_ref(), policy) } {
                    if let Ok(mut cb) = cb.lock() {
                        cb(err);
                    }
                }
            });
            if let Some(name) = unsafe { AVAudioSessionRouteChangeNotification } {
                let observer = unsafe {
                    nc.addObserverForName_object_queue_usingBlock(Some(name), None, None, &block)
                };
                observers.push(observer);
            }
        }

        {
            let cb = error_callback.clone();
            let block = RcBlock::new(move |_: NonNull<NSNotification>| {
                if let Ok(mut cb) = cb.lock() {
                    cb(Error::with_message(
                        ErrorKind::DeviceNotAvailable,
                        "audio media services were lost",
                    ));
                }
            });
            if let Some(name) = unsafe { AVAudioSessionMediaServicesWereLostNotification } {
                let observer = unsafe {
                    nc.addObserverForName_object_queue_usingBlock(Some(name), None, None, &block)
                };
                observers.push(observer);
            }
        }

        {
            let cb = error_callback.clone();
            let block = RcBlock::new(move |_: NonNull<NSNotification>| {
                if let Ok(mut cb) = cb.lock() {
                    cb(Error::with_message(
                        ErrorKind::StreamInvalidated,
                        "audio media services were reset",
                    ));
                }
            });
            if let Some(name) = unsafe { AVAudioSessionMediaServicesWereResetNotification } {
                let observer = unsafe {
                    nc.addObserverForName_object_queue_usingBlock(Some(name), None, None, &block)
                };
                observers.push(observer);
            }
        }

        Self { observers }
    }
}

impl Drop for SessionEventManager {
    fn drop(&mut self) {
        let nc = NSNotificationCenter::defaultCenter();
        for observer in &self.observers {
            unsafe { nc.removeObserver(observer.as_ref()) };
        }
    }
}
