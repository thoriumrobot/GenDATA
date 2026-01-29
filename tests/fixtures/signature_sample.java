// Test fixture for Signature String annotation placement
// This file contains various class name patterns for testing

public class SignatureSample {
    
    // Binary name with package (dotted format)
    String binaryName = "java.lang.String";
    
    // Binary name with inner class
    String innerClassName = "java.util.Map$Entry";
    
    // Internal form (slashed format)
    String internalForm = "java/lang/String";
    
    // Field descriptor
    String fieldDescriptor = "Ljava/lang/String;";
    
    // Array field descriptor
    String arrayDescriptor = "[Ljava/lang/Object;";
    
    // Primitive array descriptor
    String primitiveArray = "[I";
    
    // Method using Class.forName
    public void loadClass(String className) throws Exception {
        Class.forName(className);
    }
    
    // ClassLoader usage
    public void loadWithClassLoader(String typeName) throws Exception {
        ClassLoader loader = getClass().getClassLoader();
        loader.loadClass(typeName);
    }
    
    // Method returning class name
    public String getTypeName() {
        return "com.example.MyClass";
    }
    
    // Class.getName() usage
    public void printClassName() {
        String name = String.class.getName();
        System.out.println(name);
    }
    
    // getCanonicalName usage
    public void printCanonicalName() {
        String canonicalName = String.class.getCanonicalName();
        System.out.println(canonicalName);
    }
    
    // TypeLiteral-like pattern
    public void registerType(String fullyQualifiedName) {
        System.out.println("Registering: " + fullyQualifiedName);
    }
    
    // Non-signature string (should NOT be annotated)
    String regularString = "This is not a class name";
    
    // URL (should NOT be annotated as internal form despite slashes)
    String url = "http://example.com/path";
    
    // Field declaration only
    String pendingClassName;
}
