# Offline Voice-Controlled Fan

An embedded system that controls a DC fan using offline voice commands.
The system uses an ESP32 as the main controller, an INMP441 microphone for audio input, a TinyML model for voice command recognition, PWM for fan speed control, and an SSD1306 OLED for user feedback.

Python is used for audio data processing, model development, evaluation, and supporting tools.

---

## 1. Project Overview

The goal of this project is to develop a voice-controlled fan that can recognize a predefined set of voice commands without relying on cloud services.

The system should be able to:

* Recognize predefined voice commands offline.
* Turn the fan on and off.
* Set the fan to different speed levels.
* Control fan speed using PWM.
* Display the current fan status and speed on an SSD1306 OLED.
* Process and train the voice recognition model using Python.
* Deploy the final TinyML model on the ESP32.
* Record and analyze system data for testing and evaluation.

---

## 2. System Architecture

```text
                     ┌───────────────┐
                     │    INMP441    │
                     │  Microphone   │
                     └───────┬───────┘
                             │ I2S
                             ▼
                     ┌───────────────┐
                     │     ESP32     │
                     │               │
                     │   TinyML      │
                     │   Command     │
                     │   Processor   │
                     └───┬───────┬───┘
                         │       │
                    PWM  │       │ I2C
                         │       ▼
                         │   ┌─────────┐
                         │   │ SSD1306 │
                         │   │  OLED   │
                         │   └─────────┘
                         ▼
                     ┌─────────┐
                     │ MOSFET  │
                     └────┬────┘
                          │
                          ▼
                        FAN
```

### Python / AI Pipeline

```text
Voice Recording
      ↓
Data Collection
      ↓
Audio Preprocessing
      ↓
Feature Extraction
      ↓
Model Training
      ↓
Evaluation
      ↓
Quantization
      ↓
TinyML Deployment
      ↓
ESP32
```

---

## 3. Voice Commands

The system uses six predefined commands.

| Command            | Function                                   |
| ------------------ | ------------------------------------------ |
| `Turn On`          | Turn the fan on at the minimum level       |
| `Turn Off`         | Turn the fan off and reset the power level |
| `Set Low`          | Set fan power to 25%                       |
| `Set Medium`       | Set fan power to 50%                       |
| `Set High`         | Set fan power to 75%                       |
| `Set Max`          | Set fan power to 100%                      |
| `Unknown`          | Unknown command, do nothing                |
| `Background_noise` | Noise form environment, people, other device, etc

### Fan Control Logic

```text
Turn Off
   ↓
Power = 0%

Turn On
   ↓
Power = 25%

Set Low
   ↓
Power = 25%

Set Medium
   ↓
Power = 50%

Set High
   ↓
Power = 75%

Set Max
   ↓
Power = 100%
```

`Set High` and `Set Max` directly set the corresponding power level.
The system does not need to pass through the intermediate levels.

When the fan is turned off, the power level is reset to `0%`.

When the fan is turned on again, it starts from `25%`.

---

## 4. Hardware

Planned hardware components:

* ESP32
* INMP441 I2S microphone
* SSD1306 OLED display
* DC fan
* MOSFET module
* Battery / power source
* Voltage regulator
* Voltage divider for battery monitoring
* Supporting electronic components

### Main Interfaces

| Component          | ESP32 Interface |
| ------------------ | --------------- |
| INMP441            | I2S             |
| SSD1306            | I2C             |
| Fan control        | PWM             |
| Battery monitoring | ADC             |

---

## 5. Software

### ESP32

The ESP32 firmware will be developed using:

* ESP-IDF
* C/C++
* FreeRTOS
* GPIO
* PWM
* I2C
* I2S
* ADC

### Python

Python will be used for:

* Audio data collection
* Audio preprocessing
* Feature extraction
* Dataset management
* Model training
* Model evaluation
* Visualization
* Testing
* Supporting development tools

---

## 6. AI / TinyML Pipeline

The voice recognition system will follow this general pipeline:

```text
Microphone
    ↓
Raw Audio
    ↓
Preprocessing
    ↓
Feature Extraction
    ↓
Mel-Spectrogram / Audio Features
    ↓
Neural Network
    ↓
8-Class Classification
    ↓
Voice Command
```

The final model will be optimized and deployed on the ESP32.

The exact model architecture and preprocessing parameters will be determined during development and experimentation.

---

## 7. Team Development

The project follows a full-stack team structure.

Each member is responsible for two voice commands and works across:

* Data collection
* AI / Machine Learning
* Python
* ESP32 firmware
* Hardware
* Integration
* Testing

### Command Ownership

| Member   | Commands              |
| -------- | --------------------- |
| Member A | `Turn On` + `Low`     |
| Member B | `Turn Off` + `Medium` |
| Member C | `High` + `Max`        |

Command ownership does not mean that each member develops an independent model.

All commands will eventually be integrated into one common dataset, one final classification model, and one ESP32 firmware system.

---

## 9. Git Workflow

The `main` branch contains the integrated and reviewed version of the project.

Each member works on a separate feature branch.

```text
main
│
├── feature/member-a-on-low
├── feature/member-b-off-medium
└── feature/member-c-high-max
```

### Development Workflow

```text
Create / switch to feature branch
            ↓
        Development
            ↓
         Testing
            ↓
          Commit
            ↓
           Push
            ↓
      Pull Request
            ↓
        Code Review
            ↓
      Merge into main
```

Members should not directly push unfinished work to `main`.

---

## 10. Python Environment

Each developer uses an independent Python virtual environment.

The virtual environment is not stored in Git.

```text
.venv/
```

Python dependencies should be documented so that all team members can reproduce the development environment.

---

## 11. Testing

Testing will be performed at multiple levels.

### Unit Testing

Individual components:

* Audio preprocessing
* Feature extraction
* AI model
* PWM control
* OLED display
* Command processing

### Integration Testing

```text
Voice
 ↓
AI
 ↓
Command
 ↓
ESP32
 ↓
PWM
 ↓
Fan
```

### System Testing

The complete system will be tested using real voice commands and physical hardware.

Important measurements may include:

* Recognition accuracy
* False recognition rate
* Response time
* Fan speed response
* System stability
* Power consumption
* Battery voltage

---

## 12. Development Status

### Phase 1 — Project Setup

* [x] Create project structure
* [x] Create `.gitignore`
* [x] Initialize Git repository
* [x] Create GitHub repository
* [x] Create `main` branch
* [ ] Add team members
* [ ] Create feature branches

### Phase 2 — Hardware Prototype

* [ ] ESP32 setup
* [ ] PWM test
* [ ] Fan + MOSFET test
* [ ] INMP441 test
* [ ] SSD1306 test
* [ ] Battery monitoring

### Phase 3 — Voice Dataset

* [ ] Define recording conditions
* [ ] Record six commands
* [ ] Organize dataset
* [ ] Preprocess audio
* [ ] Analyze dataset

### Phase 4 — AI Model

* [ ] Feature extraction
* [ ] Model design
* [ ] Training
* [ ] Evaluation
* [ ] Optimization
* [ ] Quantization

### Phase 5 — ESP32 Integration

* [ ] Integrate TinyML model
* [ ] Implement command processing
* [ ] Implement fan state machine
* [ ] Implement PWM control
* [ ] Implement OLED feedback
* [ ] Integrate all components

### Phase 6 — System Testing

* [ ] Voice command testing
* [ ] Accuracy evaluation
* [ ] Response time evaluation
* [ ] Fan control testing
* [ ] Power testing
* [ ] Long-duration testing

### Phase 7 — Finalization

* [ ] Final documentation
* [ ] Final test results
* [ ] Demonstration
* [ ] Final presentation
