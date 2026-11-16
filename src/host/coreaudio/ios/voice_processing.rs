//! One `VoiceProcessingIO` unit shared by an input and an output stream.
//!
//! Selected with [`IosStreamConfig::HostOwnedVoiceProcessing`](crate::IosStreamConfig).
//!
//! The default iOS backend gives every stream its own `RemoteIO` unit, so a
//! call that records and plays opens two independent units. That works for
//! audio, but it is not how iOS expects a voice call to look. A call's voice
//! path is the `VoiceProcessingIO` unit: Apple's echo cancellation, noise
//! suppression and AGC run in it, and the system's call routing (CallKit's
//! speaker button, for one) tracks the route through it. With two plain
//! `RemoteIO` units CallKit picks the speaker, audio moves to it, and
//! `callservicesd` still reads the route back as the receiver.
//!
//! `VoiceProcessingIO` is full duplex: one unit, with the mic on its input
//! bus and the speaker on its output bus, driven by one I/O cycle (which is
//! also what lets it cancel echo). cpal's API builds input and output streams
//! separately, so this module keeps a single process-wide unit that both
//! streams attach to:
//!
//! * The unit is created when the first stream is built and disposed when the
//!   last one is dropped. It runs while at least one attached stream is
//!   playing.
//! * Both callbacks are installed once, at creation. They dispatch to a slot
//!   per direction and only `try_lock` it, so the real-time thread never
//!   blocks. With no output stream attached (or it paused) the output bus
//!   renders silence; with no input stream the mic samples are dropped.
//! * A stream asking for a format the unit was not created with recreates the
//!   unit with both formats. In practice the second stream asks for the same
//!   format as the first (mono at the session rate) and attaches as is.

use std::sync::{Arc, Mutex, Weak};

use coreaudio::audio_unit::{
    render_callback::{self, data},
    AudioUnit, Element, IOType, Scope,
};
use objc2_audio_toolbox::{kAudioOutputUnitProperty_EnableIO, kAudioUnitProperty_StreamFormat};
use objc2_core_audio_types::AudioStreamBasicDescription;

use super::session_event_manager::ErrorCallbackMutex;
use super::{extract_audio_buffer, get_device_buffer_frames, host_time_to_stream_instant};
use crate::{
    host::frames_to_duration, ChannelCount, Data, Error, ErrorKind, FrameCount, InputCallbackInfo,
    InputStreamTimestamp, OutputCallbackInfo, OutputStreamTimestamp, SampleFormat, SampleRate,
    StreamInstant, SupportedBufferSize, SupportedStreamConfig,
};

type InputFn = Box<dyn FnMut(&Data, &InputCallbackInfo) + Send + 'static>;
type OutputFn = Box<dyn FnMut(&mut Data, &OutputCallbackInfo) + Send + 'static>;

/// Which bus of the shared unit a stream is attached to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Direction {
    Input,
    Output,
}

impl Direction {
    /// The client side of this direction's bus: where its format is set.
    fn scope_element(self) -> (Scope, Element) {
        match self {
            // Mic data leaves the input bus on its output scope.
            Direction::Input => (Scope::Output, Element::Input),
            // Speaker data enters the output bus on its input scope.
            Direction::Output => (Scope::Input, Element::Output),
        }
    }
}

struct InputSlot {
    data: InputFn,
    error: ErrorCallbackMutex,
    playing: bool,
}

struct OutputSlot {
    data: OutputFn,
    error: ErrorCallbackMutex,
    playing: bool,
}

/// What the real-time callbacks dispatch to. Shared with the callbacks, so it
/// outlives every unit built on it.
#[derive(Default)]
struct Slots {
    input: Mutex<Option<InputSlot>>,
    output: Mutex<Option<OutputSlot>>,
}

impl Slots {
    fn any_playing(&self) -> bool {
        let input = self
            .input
            .lock()
            .map(|s| s.as_ref().is_some_and(|s| s.playing));
        let output = self
            .output
            .lock()
            .map(|s| s.as_ref().is_some_and(|s| s.playing));
        input.unwrap_or(false) || output.unwrap_or(false)
    }
}

struct UnitState {
    unit: AudioUnit,
    sample_format: SampleFormat,
    input_format: AudioStreamBasicDescription,
    output_format: AudioStreamBasicDescription,
    running: bool,
}

/// The process-wide unit and the slots its callbacks dispatch to.
pub(super) struct SharedUnit {
    state: Mutex<UnitState>,
    slots: Arc<Slots>,
}

/// The unit currently in use, if any stream holds it.
static SHARED: Mutex<Weak<SharedUnit>> = Mutex::new(Weak::new());

fn same_format(a: &AudioStreamBasicDescription, b: &AudioStreamBasicDescription) -> bool {
    a.mSampleRate == b.mSampleRate
        && a.mFormatID == b.mFormatID
        && a.mFormatFlags == b.mFormatFlags
        && a.mChannelsPerFrame == b.mChannelsPerFrame
        && a.mBitsPerChannel == b.mBitsPerChannel
        && a.mBytesPerFrame == b.mBytesPerFrame
}

fn lock_err() -> Error {
    Error::with_message(
        ErrorKind::StreamInvalidated,
        "voice-processing unit lock poisoned",
    )
}

/// Create a `VoiceProcessingIO` unit with both buses enabled, the given
/// client formats, and callbacks dispatching to `slots`. Not started.
fn create_unit(
    slots: &Arc<Slots>,
    sample_format: SampleFormat,
    input_format: &AudioStreamBasicDescription,
    output_format: &AudioStreamBasicDescription,
) -> Result<AudioUnit, Error> {
    let mut unit = AudioUnit::new(IOType::VoiceProcessingIO)?;
    // Bus enablement and formats are set before initialization.
    unit.uninitialize()?;

    let enable = 1u32;
    unit.set_property(
        kAudioOutputUnitProperty_EnableIO,
        Scope::Input,
        Element::Input,
        Some(&enable),
    )?;
    unit.set_property(
        kAudioOutputUnitProperty_EnableIO,
        Scope::Output,
        Element::Output,
        Some(&enable),
    )?;

    let (scope, element) = Direction::Input.scope_element();
    unit.set_property(
        kAudioUnitProperty_StreamFormat,
        scope,
        element,
        Some(input_format),
    )?;
    let (scope, element) = Direction::Output.scope_element();
    unit.set_property(
        kAudioUnitProperty_StreamFormat,
        scope,
        element,
        Some(output_format),
    )?;

    install_output_callback(
        &mut unit,
        slots.clone(),
        sample_format,
        output_format.mSampleRate,
    )?;
    install_input_callback(
        &mut unit,
        slots.clone(),
        sample_format,
        input_format.mSampleRate,
    )?;

    unit.initialize()?;
    Ok(unit)
}

fn install_output_callback(
    unit: &mut AudioUnit,
    slots: Arc<Slots>,
    sample_format: SampleFormat,
    sample_rate: f64,
) -> Result<(), Error> {
    let bytes_per_channel = sample_format.sample_size();
    let sample_rate = sample_rate as SampleRate;
    let device_buffer_frames = get_device_buffer_frames();
    type Args = render_callback::Args<data::Raw>;

    unit.set_render_callback(move |args: Args| {
        // SAFETY: CoreAudio provides a valid AudioBufferList for the callback duration.
        let (buffer, mut data) =
            unsafe { extract_audio_buffer(&args, bytes_per_channel, sample_format, false) };

        // Never block the I/O thread: if the slot is being swapped, play silence.
        let mut guard = slots.output.try_lock().ok();
        let slot = guard
            .as_mut()
            .and_then(|g| g.as_mut())
            .filter(|s| s.playing);
        let Some(slot) = slot else {
            if !buffer.mData.is_null() {
                // SAFETY: mData points to mDataByteSize writable bytes.
                unsafe {
                    std::ptr::write_bytes(buffer.mData as *mut u8, 0, buffer.mDataByteSize as usize)
                };
            }
            return Ok(());
        };

        let callback = match host_time_to_stream_instant(args.time_stamp.mHostTime) {
            Ok(cb) => cb,
            Err(err) => {
                if let Ok(mut cb) = slot.error.try_lock() {
                    cb(err);
                }
                return Err(());
            }
        };
        let delay = frames_to_duration(device_buffer_frames as FrameCount, sample_rate);
        let timestamp = OutputStreamTimestamp {
            callback,
            playback: callback + delay,
        };
        (slot.data)(&mut data, &OutputCallbackInfo { timestamp });
        Ok(())
    })?;
    Ok(())
}

fn install_input_callback(
    unit: &mut AudioUnit,
    slots: Arc<Slots>,
    sample_format: SampleFormat,
    sample_rate: f64,
) -> Result<(), Error> {
    let bytes_per_channel = sample_format.sample_size();
    let sample_rate = sample_rate as SampleRate;
    let device_buffer_frames = get_device_buffer_frames();
    type Args = render_callback::Args<data::Raw>;

    // coreaudio-rs pulls the mic samples (AudioUnitRender on the input bus)
    // before calling this, so dropping them here is all "no input stream"
    // takes.
    unit.set_input_callback(move |args: Args| {
        let mut guard = slots.input.try_lock().ok();
        let Some(slot) = guard
            .as_mut()
            .and_then(|g| g.as_mut())
            .filter(|s| s.playing)
        else {
            return Ok(());
        };

        // SAFETY: CoreAudio provides a valid AudioBufferList for the callback duration.
        let (_buffer, data) =
            unsafe { extract_audio_buffer(&args, bytes_per_channel, sample_format, true) };

        let callback = match host_time_to_stream_instant(args.time_stamp.mHostTime) {
            Ok(cb) => cb,
            Err(err) => {
                if let Ok(mut cb) = slot.error.try_lock() {
                    cb(err);
                }
                return Err(());
            }
        };
        let delay = frames_to_duration(device_buffer_frames as FrameCount, sample_rate);
        let capture = callback.checked_sub(delay).unwrap_or(StreamInstant::ZERO);
        let timestamp = InputStreamTimestamp { callback, capture };
        (slot.data)(&data, &InputCallbackInfo { timestamp });
        Ok(())
    })?;
    Ok(())
}

impl SharedUnit {
    /// The unit in use, or a new one whose `direction` bus has `format` and
    /// whose other bus has the matching mono format.
    fn get_or_create(
        direction: Direction,
        format: &AudioStreamBasicDescription,
        sample_format: SampleFormat,
    ) -> Result<Arc<SharedUnit>, Error> {
        let mut current = SHARED.lock().map_err(|_| lock_err())?;
        if let Some(shared) = current.upgrade() {
            return Ok(shared);
        }

        // The other direction is not known yet. Assume mono at the same rate,
        // which is what a voice call asks for; `attach` recreates the unit if
        // the other stream turns out to want something else.
        let mut other = *format;
        let bytes_per_sample = sample_format.sample_size() as u32;
        other.mChannelsPerFrame = 1;
        other.mBytesPerFrame = bytes_per_sample;
        other.mBytesPerPacket = bytes_per_sample;
        let (input_format, output_format) = match direction {
            Direction::Input => (*format, other),
            Direction::Output => (other, *format),
        };

        let slots = Arc::new(Slots::default());
        let unit = create_unit(&slots, sample_format, &input_format, &output_format)?;
        let shared = Arc::new(SharedUnit {
            state: Mutex::new(UnitState {
                unit,
                sample_format,
                input_format,
                output_format,
                running: false,
            }),
            slots,
        });
        *current = Arc::downgrade(&shared);
        Ok(shared)
    }

    /// Make sure `direction`'s bus carries `format`, recreating the unit if
    /// it does not, and return the format the unit granted.
    fn ensure_format(
        &self,
        direction: Direction,
        format: &AudioStreamBasicDescription,
        sample_format: SampleFormat,
    ) -> Result<AudioStreamBasicDescription, Error> {
        let mut state = self.state.lock().map_err(|_| lock_err())?;
        let current = match direction {
            Direction::Input => &state.input_format,
            Direction::Output => &state.output_format,
        };
        if !same_format(current, format) || state.sample_format != sample_format {
            let (input_format, output_format) = match direction {
                Direction::Input => (*format, state.output_format),
                Direction::Output => (state.input_format, *format),
            };
            let was_running = state.running;
            if was_running {
                state.unit.stop()?;
                state.running = false;
            }
            // The old unit is disposed when replaced. Its callbacks hold their
            // own `Arc<Slots>`, so the slots survive it.
            state.unit = create_unit(&self.slots, sample_format, &input_format, &output_format)?;
            state.sample_format = sample_format;
            state.input_format = input_format;
            state.output_format = output_format;
            if was_running {
                state.unit.start()?;
                state.running = true;
            }
        }
        let (scope, element) = direction.scope_element();
        let granted: AudioStreamBasicDescription =
            state
                .unit
                .get_property(kAudioUnitProperty_StreamFormat, scope, element)?;
        Ok(granted)
    }

    /// Start the unit if any stream is playing, stop it if none is.
    fn sync_running(&self) -> Result<(), Error> {
        let mut state = self.state.lock().map_err(|_| lock_err())?;
        let want = self.slots.any_playing();
        if want && !state.running {
            state.unit.start()?;
            state.running = true;
        } else if !want && state.running {
            state.unit.stop()?;
            state.running = false;
        }
        Ok(())
    }
}

/// A stream's hold on the shared unit. Dropping it detaches the stream; the
/// unit itself goes when the last attachment does.
pub(super) struct Attachment {
    shared: Arc<SharedUnit>,
    direction: Direction,
}

impl Attachment {
    fn set_playing(&self, playing: bool) -> Result<(), Error> {
        match self.direction {
            Direction::Input => {
                if let Some(slot) = self
                    .shared
                    .slots
                    .input
                    .lock()
                    .map_err(|_| lock_err())?
                    .as_mut()
                {
                    slot.playing = playing;
                }
            }
            Direction::Output => {
                if let Some(slot) = self
                    .shared
                    .slots
                    .output
                    .lock()
                    .map_err(|_| lock_err())?
                    .as_mut()
                {
                    slot.playing = playing;
                }
            }
        }
        self.shared.sync_running()
    }

    pub(super) fn play(&self) -> Result<(), Error> {
        self.set_playing(true)
    }

    pub(super) fn pause(&self) -> Result<(), Error> {
        self.set_playing(false)
    }
}

impl Drop for Attachment {
    fn drop(&mut self) {
        match self.direction {
            Direction::Input => {
                if let Ok(mut slot) = self.shared.slots.input.lock() {
                    *slot = None;
                }
            }
            Direction::Output => {
                if let Ok(mut slot) = self.shared.slots.output.lock() {
                    *slot = None;
                }
            }
        }
        // Stop the unit if the other stream is not playing either. If this was
        // the last attachment, dropping `shared` right after disposes it.
        let _ = self.shared.sync_running();
    }
}

fn granted_config(
    granted: &AudioStreamBasicDescription,
    sample_format: SampleFormat,
) -> SupportedStreamConfig {
    let frames = get_device_buffer_frames() as FrameCount;
    SupportedStreamConfig::new(
        granted.mChannelsPerFrame as ChannelCount,
        granted.mSampleRate as SampleRate,
        SupportedBufferSize::Range {
            min: frames,
            max: frames,
        },
        sample_format,
    )
}

/// Attach an input stream to the shared unit, starting it.
pub(super) fn attach_input(
    format: &AudioStreamBasicDescription,
    sample_format: SampleFormat,
    data: InputFn,
    error: ErrorCallbackMutex,
) -> Result<(Attachment, SupportedStreamConfig), Error> {
    let shared = SharedUnit::get_or_create(Direction::Input, format, sample_format)?;
    let granted = shared.ensure_format(Direction::Input, format, sample_format)?;
    {
        let mut slot = shared.slots.input.lock().map_err(|_| lock_err())?;
        if slot.is_some() {
            return Err(Error::with_message(
                ErrorKind::UnsupportedConfig,
                "the shared VoiceProcessingIO unit already has an input stream",
            ));
        }
        *slot = Some(InputSlot {
            data,
            error,
            playing: true,
        });
    }
    let attachment = Attachment {
        shared,
        direction: Direction::Input,
    };
    attachment.shared.sync_running()?;
    Ok((attachment, granted_config(&granted, sample_format)))
}

/// Attach an output stream to the shared unit, starting it.
pub(super) fn attach_output(
    format: &AudioStreamBasicDescription,
    sample_format: SampleFormat,
    data: OutputFn,
    error: ErrorCallbackMutex,
) -> Result<(Attachment, SupportedStreamConfig), Error> {
    let shared = SharedUnit::get_or_create(Direction::Output, format, sample_format)?;
    let granted = shared.ensure_format(Direction::Output, format, sample_format)?;
    {
        let mut slot = shared.slots.output.lock().map_err(|_| lock_err())?;
        if slot.is_some() {
            return Err(Error::with_message(
                ErrorKind::UnsupportedConfig,
                "the shared VoiceProcessingIO unit already has an output stream",
            ));
        }
        *slot = Some(OutputSlot {
            data,
            error,
            playing: true,
        });
    }
    let attachment = Attachment {
        shared,
        direction: Direction::Output,
    };
    attachment.shared.sync_running()?;
    Ok((attachment, granted_config(&granted, sample_format)))
}
