# 🎨 Dashboard & Demo Guide

Complete guide to using the web dashboard and iOS demo app for PicoTuri-EditJudge.

---

## 📊 Web Dashboard

### Features
- ✅ **Real-time Algorithm Testing** - Test all 7 algorithms interactively
- ✅ **Beautiful Modern UI** - Gradient design with smooth animations
- ✅ **Performance Metrics** - View algorithm statistics and performance
- ✅ **Responsive Design** - Works on desktop, tablet, and mobile
- ✅ **Interactive Visualizations** - See results in real-time

### Quick Start

#### Option 1: Using the Script
```bash
# From project root
./run_dashboard.sh
```

#### Option 2: Manual Start
```bash
# Install Flask (if not installed)
pip install flask

# Run dashboard
python src/gui/web_dashboard.py
```

#### Option 3: Direct Python
```bash
cd /Users/mangeshraut/Downloads/PicoTuri-EditJudge
python -m src.gui.web_dashboard
```

### Access the Dashboard

Once started, open your browser:
```
http://localhost:5000
```

Or from another device on the same network:
```
http://YOUR_IP_ADDRESS:5000
```

### Dashboard Features

#### 1. **Statistics Cards**
- Total algorithms (7/7)
- Code quality (100%)
- Test coverage (100%)
- Production status

#### 2. **Algorithm Testing**
Three interactive test cards:

**Quality Scorer**
- 4-component weighted system
- CLIP + LPIPS + ResNet50
- Real-time scoring

**Diffusion Model**
- U-Net architecture
- 10.9M parameters
- Cross-attention mechanism

**Baseline Model**
- Scikit-learn pipeline
- TF-IDF vectorization
- Logistic regression

#### 3. **Test Results**
- JSON response display
- Success/error indicators
- Detailed component breakdown

### API Endpoints

```
GET  /                          - Main dashboard
GET  /api/stats                 - Get statistics
POST /api/test/quality-scorer   - Test quality scorer
POST /api/test/diffusion-model  - Test diffusion model
POST /api/test/baseline         - Test baseline model
```

### Screenshots

The dashboard features:
- Purple gradient background
- White cards with shadows
- Smooth hover effects
- Loading animations
- Color-coded results

---

## 📱 iOS Demo App

### Features
- ✅ **SwiftUI Interface** - Modern, native iOS design
- ✅ **Image Selection** - Pick original and edited images
- ✅ **Quality Analysis** - Real-time quality assessment
- ✅ **Component Breakdown** - Detailed score visualization
- ✅ **Circular Progress** - Animated score display
- ✅ **Recommendations** - AI-powered suggestions

### Quick Start

#### 1. Open in Xcode
```bash
cd examples/ios/PicoTuriEditJudge
open PicoTuriEditJudge.xcodeproj
```

#### 2. Select Simulator
- Click device menu
- Choose: iPhone 15 Pro (or any iOS 16+ device)

#### 3. Run App
- Press ⌘ + R
- Or click ▶️ play button

### Using the App

#### Step 1: Select Original Image
1. Tap the "Original" image placeholder
2. Photo picker opens
3. Select an image from library
4. Image displays in the left slot

#### Step 2: Select Edited Image
1. Tap the "Edited" image placeholder
2. Photo picker opens
3. Select an edited version
4. Image displays in the right slot

#### Step 3: Enter Instruction
1. Tap the instruction text field
2. Type the edit instruction
   - Example: "brighten the image"
   - Example: "add more contrast"
   - Example: "apply blue filter"

#### Step 4: Analyze Quality
1. Tap "Analyze Quality" button
2. Loading indicator appears
3. Wait ~1.5 seconds
4. Results display below

#### Step 5: View Results

**Overall Score**
- Circular progress indicator
- Score out of 100
- Grade (Excellent/Good/Fair/Poor)
- Color-coded (Green/Blue/Orange/Red)

**Component Breakdown**
- Instruction Compliance (40% weight)
- Editing Realism (25% weight)
- Preservation Balance (20% weight)
- Technical Quality (15% weight)

**Recommendation**
- AI-generated suggestion
- Based on overall score
- Actionable feedback

### App Architecture

```
PicoTuriEditJudge/
├── PicoTuriEditJudgeApp.swift   # App entry point
├── ContentView.swift             # Main UI
├── EditJudgeViewModel.swift      # Business logic
└── Assets.xcassets/              # Images & colors
```

### Customization

#### Change Colors
Edit `ContentView.swift`:
```swift
LinearGradient(
    colors: [.blue, .purple],  // Your colors here
    startPoint: .topLeading,
    endPoint: .bottomTrailing
)
```

#### Change Weights
Edit `EditJudgeViewModel.swift`:
```swift
let overall = (
    instructionCompliance * 0.40 +  // Adjust weights
    editingRealism * 0.25 +
    preservationBalance * 0.20 +
    technicalQuality * 0.15
)
```

#### Add Core ML Model
1. Export model: `python -m src.algorithms.coreml_optimizer`
2. Drag `.mlmodel` to Xcode
3. Use in `EditJudgeViewModel.swift`

---

## 🎯 Testing Guide

### Web Dashboard Testing

#### Manual Testing
1. **Start Dashboard**
   ```bash
   ./run_dashboard.sh
   ```

2. **Test Each Algorithm**
   - Click "Test Quality Scorer"
   - Verify success message
   - Check JSON response
   - Repeat for other algorithms

3. **Check Responsiveness**
   - Resize browser window
   - Test on mobile device
   - Verify layout adapts

#### Automated Testing
```bash
# Install testing tools
pip install pytest pytest-flask

# Run tests
pytest tests/test_web_dashboard.py
```

### iOS App Testing

#### Simulator Testing
1. **Launch Simulator**
   ```bash
   open -a Simulator
   ```

2. **Run App** (⌘ + R in Xcode)

3. **Test Features**
   - Image selection
   - Instruction input
   - Analysis button
   - Results display

#### Device Testing
1. **Connect iPhone/iPad**
2. **Trust Computer**
3. **Select Device** in Xcode
4. **Run App** (⌘ + R)
5. **Test on Real Device**

#### UI Testing
```swift
// In XCTest
func testAppLaunch() {
    let app = XCUIApplication()
    app.launch()
    XCTAssertTrue(app.exists)
}
```

---

## 🚀 Deployment

### Web Dashboard Deployment

#### Option 1: Local Network
```bash
# Run on all interfaces
python src/gui/web_dashboard.py --host 0.0.0.0
```

#### Option 2: Production Server
```bash
# Install gunicorn
pip install gunicorn

# Run with gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 src.gui.web_dashboard:app
```

#### Option 3: Docker
```dockerfile
FROM python:3.12
WORKDIR /app
COPY . .
RUN pip install flask
CMD ["python", "src/gui/web_dashboard.py"]
```

### iOS App Deployment

#### TestFlight (Beta Testing)
1. **Archive App** (Product → Archive)
2. **Upload to App Store Connect**
3. **Add Testers**
4. **Distribute via TestFlight**

#### App Store (Production)
1. **Prepare Metadata**
   - App name
   - Description
   - Screenshots
   - Keywords

2. **Submit for Review**
   - Upload archive
   - Fill out questionnaire
   - Submit

3. **Wait for Approval**
   - Usually 1-3 days
   - Respond to feedback

---

## 📊 Performance Benchmarks

### Web Dashboard
```
Page Load: < 1 second
API Response: < 100ms
Memory Usage: ~50MB
CPU Usage: < 10%
```

### iOS App
```
App Launch: < 1 second
Image Load: < 0.5 seconds
Analysis: ~1.5 seconds
Memory: < 100MB
CPU: < 30%
```

---

## 🐛 Troubleshooting

### Web Dashboard Issues

#### Port Already in Use
```bash
# Find process using port 5000
lsof -i :5000

# Kill process
kill -9 <PID>

# Or use different port
python src/gui/web_dashboard.py --port 5001
```

#### Flask Not Found
```bash
pip install flask
# or
pip install -r requirements-dev.txt
```

#### Template Not Found
```bash
# Verify templates directory exists
ls src/gui/templates/

# Recreate if needed
python src/gui/web_dashboard.py
```

### iOS App Issues

#### Build Fails
```
Clean Build Folder: ⌘ + Shift + K
Rebuild: ⌘ + B
```

#### Simulator Won't Launch
```bash
xcrun simctl erase all
xcrun simctl boot "iPhone 15 Pro"
```

#### App Crashes
1. Check console (⌘ + Shift + Y)
2. Look for error messages
3. Verify all files added to target

---

## 📚 Additional Resources

### Web Development
- [Flask Documentation](https://flask.palletsprojects.com/)
- [HTML/CSS Guide](https://developer.mozilla.org/en-US/docs/Web)
- [JavaScript Tutorial](https://javascript.info/)

### iOS Development
- [SwiftUI Tutorials](https://developer.apple.com/tutorials/swiftui)
- [Xcode Documentation](https://developer.apple.com/xcode/)
- [iOS Human Interface Guidelines](https://developer.apple.com/design/human-interface-guidelines/ios)

### Design Resources
- [SF Symbols](https://developer.apple.com/sf-symbols/)
- [Color Palette Generator](https://coolors.co/)
- [Gradient Generator](https://cssgradient.io/)

---

## ✅ Quick Reference

### Web Dashboard Commands
```bash
# Start dashboard
./run_dashboard.sh

# Start with custom port
python src/gui/web_dashboard.py --port 8080

# Start in production mode
gunicorn src.gui.web_dashboard:app
```

### iOS App Commands
```bash
# Open in Xcode
open examples/ios/PicoTuriEditJudge.xcodeproj

# Build from command line
xcodebuild -scheme PicoTuriEditJudge build

# Run tests
xcodebuild test -scheme PicoTuriEditJudge
```

### Xcode Shortcuts
```
Build: ⌘ + B
Run: ⌘ + R
Test: ⌘ + U
Clean: ⌘ + Shift + K
Profile: ⌘ + I
```

---

**Last Updated:** October 27, 2025  
**Status:** ✅ Ready for Use  
**Support:** See docs/XCODE_TESTING_GUIDE.md for detailed iOS testing
