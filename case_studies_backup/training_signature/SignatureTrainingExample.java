// Signature String Training Example
// This file demonstrates the pattern for training:
// Entry point methods are annotated, internal variables are not.
// The model learns to add annotations to reduce warnings.

import org.checkerframework.checker.signature.qual.BinaryName;
import org.checkerframework.checker.signature.qual.FullyQualifiedName;
import org.checkerframework.checker.signature.qual.ClassGetName;
import org.checkerframework.checker.signature.qual.InternalForm;

public class SignatureTrainingExample {
    
    // === ENTRY POINTS (annotated) ===
    // These method parameters are annotated, creating requirements for callers
    
    public Class<?> loadClass(@BinaryName String className) throws ClassNotFoundException {
        return Class.forName(className);
    }
    
    public void registerType(@FullyQualifiedName String typeName) {
        System.out.println("Registering: " + typeName);
    }
    
    public void processDescriptor(@InternalForm String descriptor) {
        System.out.println("Processing: " + descriptor);
    }
    
    // === WARNING SITES (unannotated) ===
    // These internal variables are NOT annotated.
    // Passing them to annotated methods creates warnings.
    // The model should learn to add appropriate annotations to fix warnings.
    
    // WARNING 1: Unannotated field passed to annotated method
    private String className1 = "java.lang.String";
    
    public void example1() throws ClassNotFoundException {
        // This line creates a warning: className1 is unannotated
        loadClass(className1);  // WARNING: [argument] incompatible
        // FIX: Add @BinaryName to className1 field
    }
    
    // WARNING 2: Unannotated local variable
    public void example2() throws ClassNotFoundException {
        String className2 = "java.util.ArrayList";
        loadClass(className2);  // WARNING: [argument] incompatible
        // FIX: Add @BinaryName to className2
    }
    
    // WARNING 3: Unannotated method return value
    public String getTypeName() {
        return "com.example.MyClass";
    }
    
    public void example3() {
        registerType(getTypeName());  // WARNING: [argument] incompatible
        // FIX: Add @FullyQualifiedName to getTypeName() return type
    }
    
    // WARNING 4: Using Class.getName() result
    public void example4() {
        String name = String.class.getName();  // Returns @ClassGetName
        // Different annotation type mismatch
        registerType(name);  // WARNING: @ClassGetName vs @FullyQualifiedName
    }
    
    // WARNING 5: String concatenation with class names
    public void example5() throws ClassNotFoundException {
        String pkg = "java.lang";
        String simple = "Integer";
        String fullName = pkg + "." + simple;
        loadClass(fullName);  // WARNING: [argument] incompatible
        // FIX: Add @BinaryName to fullName
    }
    
    // === CORRECT EXAMPLES (no warnings) ===
    // These show the pattern after fix is applied
    
    @BinaryName String correctClassName = "java.lang.String";
    
    public void correctExample() throws ClassNotFoundException {
        loadClass(correctClassName);  // No warning
    }
    
    public @FullyQualifiedName String correctGetTypeName() {
        return "com.example.MyClass";
    }
    
    public void correctExample2() {
        registerType(correctGetTypeName());  // No warning
    }
}
