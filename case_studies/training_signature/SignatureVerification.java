// Signature Verification - Shows warning reduction by adding annotations
// Run this file with and without the @BinaryName annotations to verify

import org.checkerframework.checker.signature.qual.BinaryName;
import org.checkerframework.checker.signature.qual.FullyQualifiedName;

public class SignatureVerification {
    
    // Entry points (annotated)
    public Class<?> loadClass(@BinaryName String className) throws ClassNotFoundException {
        return Class.forName(className);
    }
    
    public void registerType(@FullyQualifiedName String typeName) {
        System.out.println(typeName);
    }
    
    // BEFORE: Unannotated - creates warning
    // String className1 = "java.lang.String";
    
    // AFTER: Annotated - no warning
    @BinaryName String className1 = "java.lang.String";
    
    // BEFORE: Unannotated return type - creates warning
    // public String getTypeName() { return "com.example.MyClass"; }
    
    // AFTER: Annotated return type - no warning
    public @FullyQualifiedName String getTypeName() { 
        return "com.example.MyClass"; 
    }
    
    public void test() throws ClassNotFoundException {
        loadClass(className1);       // No warning after adding @BinaryName
        registerType(getTypeName()); // No warning after adding @FullyQualifiedName
    }
}
