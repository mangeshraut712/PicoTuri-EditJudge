import SwiftUI

@main
struct EditJudgeDemoApp: App {
    @StateObject private var model = ModelController()

    var body: some Scene {
        WindowGroup {
            ContentView(model: model)
        }
    }
}
