# === TensorFlow Lite GPU Delegate ===
-keep class org.tensorflow.lite.** { *; }
-dontwarn org.tensorflow.lite.**

-keep class org.tensorflow.lite.gpu.** { *; }
-dontwarn org.tensorflow.lite.gpu.**

# === Flutter and AndroidX Plugins ===
-keep class io.flutter.** { *; }
-dontwarn io.flutter.**

-keep class androidx.lifecycle.** { *; }
-dontwarn androidx.lifecycle.**

-keep class androidx.camera.** { *; }
-dontwarn androidx.camera.**

-keep class io.flutter.plugins.** { *; }
-dontwarn io.flutter.plugins.**

# Ignore warnings for unused ML Kit language-specific classes
-dontwarn com.google.mlkit.vision.text.chinese.**
-dontwarn com.google.mlkit.vision.text.devanagari.**
-dontwarn com.google.mlkit.vision.text.japanese.**
-dontwarn com.google.mlkit.vision.text.korean.**

# Keep only Latin text recognition and core classes
-keep class com.google.mlkit.vision.text.TextRecognizer { *; }
-keep class com.google.mlkit.vision.text.latin.** { *; }

# Do NOT keep language-specific recognizers (let them be stripped safely)
-assumenosideeffects class com.google.mlkit.vision.text.chinese.** { *; }
-assumenosideeffects class com.google.mlkit.vision.text.devanagari.** { *; }
-assumenosideeffects class com.google.mlkit.vision.text.japanese.** { *; }
-assumenosideeffects class com.google.mlkit.vision.text.korean.** { *; }
