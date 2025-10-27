# 📱 Xcode Testing Guide - PicoTuri-EditJudge iOS Demo

Complete guide to setting up, building, and testing the iOS demo app in Xcode.

---

## 🎯 Quick Start

### Prerequisites
- **macOS** 13.0 or later
- **Xcode** 15.0 or later
- **iOS Simulator** or physical iOS device (iOS 16.0+)
- **Apple Developer Account** (for device testing)

---

## 📦 Step 1: Create Xcode Project

### Option A: Using Xcode GUI

1. **Open Xcode**
   ```
   Launch Xcode from Applications
   ```

2. **Create New Project**
   - Click "Create a new Xcode project"
   - Select **iOS** → **App**
   - Click **Next**

3. **Configure Project**
   ```
   Product Name: PicoTuriEditJudge
   Team: [Your Team]
   Organization Identifier: com.yourcompany
   Interface: SwiftUI
   Language: Swift
   ```

4. **Save Location**
   ```
   Navigate to: PicoTuri-EditJudge/examples/ios/
   Click "Create"
   ```

### Option B: Using Command Line

```bash
cd /Users/mangeshraut/Downloads/PicoTuri-EditJudge/examples/ios

# Create Xcode project structure
mkdir -p PicoTuriEditJudge.xcodeproj
```

---

## 📝 Step 2: Add Source Files

### 1. Add Swift Files to Project

In Xcode:
1. **Right-click** on project navigator
2. Select **Add Files to "PicoTuriEditJudge"**
3. Navigate to `examples/ios/PicoTuriEditJudge/`
4. Select all `.swift` files:
   - `PicoTuriEditJudgeApp.swift`
   - `ContentView.swift`
   - `EditJudgeViewModel.swift`
5. Check **"Copy items if needed"**
6. Click **Add**

### 2. Verify File Structure

Your project should look like:
```
PicoTuriEditJudge/
├── PicoTuriEditJudgeApp.swift
├── ContentView.swift
├── EditJudgeViewModel.swift
├── Assets.xcassets/
└── Info.plist
```

---

## ⚙️ Step 3: Configure Project Settings

### 1. General Settings

1. Select **project** in navigator
2. Go to **General** tab
3. Configure:
   ```
   Display Name: PicoTuri EditJudge
   Bundle Identifier: com.yourcompany.PicoTuriEditJudge
   Version: 1.0
   Build: 1
   
   Deployment Info:
   - iOS: 16.0
   - iPhone and iPad
   - Portrait orientation
   ```

### 2. Signing & Capabilities

1. Go to **Signing & Capabilities** tab
2. Enable **Automatically manage signing**
3. Select your **Team**
4. Add capabilities (if needed):
   - Photo Library (for image selection)

### 3. Info.plist Configuration

Add privacy descriptions:
```xml
<key>NSPhotoLibraryUsageDescription</key>
<string>We need access to your photo library to select images for quality assessment.</string>

<key>NSCameraUsageDescription</key>
<string>We need access to your camera to take photos for quality assessment.</string>
```

---

## 🔨 Step 4: Build the Project

### Using Xcode GUI

1. **Select Simulator**
   - Click device menu (top toolbar)
   - Choose: **iPhone 15 Pro** (or any iOS 16+ device)

2. **Build Project**
   - Press **⌘ + B** (Command + B)
   - Or: Product → Build

3. **Check for Errors**
   - View build log in **Report Navigator** (⌘ + 9)
   - Fix any compilation errors

### Using Command Line

```bash
# Navigate to project directory
cd /Users/mangeshraut/Downloads/PicoTuri-EditJudge/examples/ios

# Build for simulator
xcodebuild -scheme PicoTuriEditJudge \
  -destination 'platform=iOS Simulator,name=iPhone 15 Pro' \
  build

# Build for device
xcodebuild -scheme PicoTuriEditJudge \
  -destination 'generic/platform=iOS' \
  build
```

---

## ▶️ Step 5: Run the App

### On iOS Simulator

1. **Select Simulator**
   ```
   Device menu → iPhone 15 Pro (or your choice)
   ```

2. **Run App**
   - Press **⌘ + R** (Command + R)
   - Or: Product → Run
   - Or: Click ▶️ play button

3. **Wait for Launch**
   - Simulator will boot (if not running)
   - App will install and launch
   - Should see main dashboard

### On Physical Device

1. **Connect Device**
   - Connect iPhone/iPad via USB
   - Trust computer on device

2. **Select Device**
   ```
   Device menu → Your iPhone/iPad
   ```

3. **Run App**
   - Press **⌘ + R**
   - App will install and launch

4. **Trust Developer** (first time)
   - Settings → General → VPN & Device Management
   - Trust your developer certificate

---

## 🧪 Step 6: Test the App

### Manual Testing Checklist

#### 1. **UI Testing**
- [ ] App launches successfully
- [ ] Main dashboard displays correctly
- [ ] All UI elements are visible
- [ ] Buttons are responsive
- [ ] Navigation works smoothly

#### 2. **Image Selection**
- [ ] Tap "Original" image placeholder
- [ ] Photo picker opens
- [ ] Select an image
- [ ] Image displays correctly
- [ ] Repeat for "Edited" image

#### 3. **Instruction Input**
- [ ] Tap instruction text field
- [ ] Keyboard appears
- [ ] Type: "brighten the image"
- [ ] Text displays correctly

#### 4. **Quality Analysis**
- [ ] "Analyze Quality" button is enabled
- [ ] Tap analyze button
- [ ] Loading indicator shows
- [ ] Results appear after ~1.5 seconds
- [ ] Overall score displays (0-100)
- [ ] Component breakdown shows
- [ ] Grade displays (Excellent/Good/Fair/Poor)
- [ ] Recommendation text appears

#### 5. **Results Display**
- [ ] Circular progress indicator animates
- [ ] Component scores show with progress bars
- [ ] All percentages are correct
- [ ] Colors are appropriate
- [ ] Layout is clean and readable

#### 6. **Algorithm Status**
- [ ] Status badges display at bottom
- [ ] All show "Ready" or "Active"
- [ ] Icons are correct

### Automated Testing

Create test file: `PicoTuriEditJudgeTests.swift`

```swift
import XCTest
@testable import PicoTuriEditJudge

class PicoTuriEditJudgeTests: XCTestCase {
    
    func testViewModelInitialization() {
        let viewModel = EditJudgeViewModel()
        XCTAssertEqual(viewModel.overallScore, 0.0)
        XCTAssertFalse(viewModel.hasResults)
    }
    
    func testAnalyzeImages() {
        let viewModel = EditJudgeViewModel()
        let testImage = UIImage(systemName: "photo")!
        
        viewModel.analyzeImages(
            original: testImage,
            edited: testImage,
            instruction: "test"
        )
        
        XCTAssertTrue(viewModel.hasResults)
        XCTAssertGreaterThan(viewModel.overallScore, 0.0)
        XCTAssertFalse(viewModel.grade.isEmpty)
    }
}
```

Run tests:
```
⌘ + U (Command + U)
or: Product → Test
```

---

## 🐛 Step 7: Debugging

### Common Issues & Solutions

#### Issue 1: Build Fails
```
Error: No such module 'SwiftUI'
```
**Solution:**
- Ensure iOS Deployment Target is 16.0+
- Clean build folder: ⌘ + Shift + K
- Rebuild: ⌘ + B

#### Issue 2: Simulator Won't Launch
```
Error: Unable to boot device
```
**Solution:**
```bash
# Reset simulator
xcrun simctl erase all
xcrun simctl boot "iPhone 15 Pro"
```

#### Issue 3: App Crashes on Launch
**Solution:**
1. Check console output (⌘ + Shift + Y)
2. Look for error messages
3. Verify all files are added to target
4. Check Info.plist configuration

#### Issue 4: Images Won't Load
```
Error: Photo library access denied
```
**Solution:**
- Add NSPhotoLibraryUsageDescription to Info.plist
- Reset privacy settings in simulator:
  ```
  Settings → General → Reset → Reset Location & Privacy
  ```

### Debug Console

View logs:
1. Open **Debug Area**: ⌘ + Shift + Y
2. Check console output
3. Look for print statements and errors

---

## 📊 Step 8: Performance Testing

### Using Instruments

1. **Profile App**
   ```
   Product → Profile (⌘ + I)
   ```

2. **Select Instrument**
   - Time Profiler (CPU usage)
   - Allocations (Memory usage)
   - Leaks (Memory leaks)

3. **Record Session**
   - Click record button
   - Use app normally
   - Stop recording
   - Analyze results

### Performance Metrics

Expected performance:
```
App Launch: < 1 second
Image Load: < 0.5 seconds
Analysis: ~1.5 seconds (simulated)
Memory: < 100 MB
CPU: < 30% average
```

---

## 📱 Step 9: UI Testing

### Create UI Test

File: `PicoTuriEditJudgeUITests.swift`

```swift
import XCTest

class PicoTuriEditJudgeUITests: XCTestCase {
    
    func testAppLaunch() {
        let app = XCUIApplication()
        app.launch()
        
        // Verify main elements exist
        XCTAssertTrue(app.staticTexts["Image Edit Quality Assessment"].exists)
        XCTAssertTrue(app.buttons["Analyze Quality"].exists)
    }
    
    func testImageSelection() {
        let app = XCUIApplication()
        app.launch()
        
        // Tap original image placeholder
        app.otherElements["Original"].tap()
        
        // Verify photo picker appears
        // (Note: Actual photo picker testing requires additional setup)
    }
}
```

Run UI tests:
```
⌘ + U (Command + U)
```

---

## 🎨 Step 10: Customization

### Change App Icon

1. **Create Icon Set**
   - Create icons: 1024x1024, 180x180, 120x120, etc.
   - Use online tool: https://appicon.co

2. **Add to Project**
   - Open `Assets.xcassets`
   - Click `AppIcon`
   - Drag icons to appropriate slots

### Change Color Scheme

Edit `ContentView.swift`:
```swift
// Change gradient colors
LinearGradient(
    colors: [.blue, .purple],  // Change these
    startPoint: .topLeading,
    endPoint: .bottomTrailing
)
```

### Add Dark Mode Support

```swift
@Environment(\.colorScheme) var colorScheme

var backgroundColor: Color {
    colorScheme == .dark ? .black : .white
}
```

---

## 📦 Step 11: Archive & Distribution

### Create Archive

1. **Select Device**
   ```
   Device menu → Any iOS Device (arm64)
   ```

2. **Archive**
   ```
   Product → Archive
   ```

3. **Wait for Build**
   - Build completes
   - Organizer window opens

### Distribute App

1. **In Organizer**
   - Select archive
   - Click **Distribute App**

2. **Choose Method**
   - App Store Connect (for App Store)
   - Ad Hoc (for testing)
   - Enterprise (for internal)
   - Development (for personal)

3. **Follow Wizard**
   - Sign with certificate
   - Upload to App Store Connect
   - Or export IPA file

---

## 🔍 Step 12: Advanced Features

### Add Core ML Model

1. **Export Model**
   ```bash
   python -m src.algorithms.coreml_optimizer
   ```

2. **Add to Xcode**
   - Drag `.mlmodel` file to project
   - Check "Copy items if needed"

3. **Use in Code**
   ```swift
   import CoreML
   
   let model = try? PicoTuriQualityScorer()
   let prediction = try? model.prediction(input: input)
   ```

### Add Networking

```swift
import Foundation

class APIService {
    func analyzeImage(image: UIImage) async throws -> QualityResult {
        // Call backend API
        let url = URL(string: "https://api.example.com/analyze")!
        // ... implementation
    }
}
```

---

## 📚 Additional Resources

### Documentation
- [Swift Documentation](https://swift.org/documentation/)
- [SwiftUI Tutorials](https://developer.apple.com/tutorials/swiftui)
- [Core ML Guide](https://developer.apple.com/documentation/coreml)

### Tools
- [SF Symbols](https://developer.apple.com/sf-symbols/) - Icons
- [AppIcon Generator](https://appicon.co) - Icon creation
- [TestFlight](https://developer.apple.com/testflight/) - Beta testing

### Community
- [Swift Forums](https://forums.swift.org)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/swiftui)
- [Apple Developer Forums](https://developer.apple.com/forums/)

---

## ✅ Testing Checklist

### Pre-Release
- [ ] All tests pass
- [ ] No compiler warnings
- [ ] App runs on simulator
- [ ] App runs on device
- [ ] All features work
- [ ] UI looks correct
- [ ] Performance is acceptable
- [ ] No memory leaks
- [ ] Privacy descriptions added
- [ ] App icon added

### Release
- [ ] Version number updated
- [ ] Build number incremented
- [ ] Archive created successfully
- [ ] App signed correctly
- [ ] Screenshots prepared
- [ ] App Store description ready
- [ ] Privacy policy prepared
- [ ] TestFlight testing complete

---

## 🎯 Quick Commands Reference

```bash
# Build
⌘ + B

# Run
⌘ + R

# Test
⌘ + U

# Clean
⌘ + Shift + K

# Profile
⌘ + I

# Archive
Product → Archive

# Show/Hide Debug Area
⌘ + Shift + Y

# Show/Hide Navigator
⌘ + 0

# Show/Hide Inspector
⌘ + Option + 0
```

---

## 🚀 Next Steps

1. **Integrate Real Core ML Model**
   - Export trained model
   - Add to Xcode project
   - Replace simulated analysis

2. **Add More Features**
   - Image editing tools
   - History of analyses
   - Share results
   - Export reports

3. **Improve UI/UX**
   - Add animations
   - Improve transitions
   - Add haptic feedback
   - Support iPad layout

4. **Deploy to App Store**
   - Create App Store listing
   - Submit for review
   - Launch app

---

**Last Updated:** October 27, 2025  
**Xcode Version:** 15.0+  
**iOS Version:** 16.0+  
**Status:** ✅ Ready for Testing
