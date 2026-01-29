// SQL Quotes Training Example
// This file demonstrates the pattern for training: 
// Entry point methods are annotated, internal variables are not.
// The model learns to add annotations to reduce warnings.

import org.checkerframework.checker.sqlquotes.qual.SqlEvenQuotes;
import org.checkerframework.checker.sqlquotes.qual.SqlOddQuotes;
import org.checkerframework.checker.sqlquotes.qual.SqlQuotesUnknown;

public class SqlQuotesTrainingExample {
    
    // === ENTRY POINTS (annotated) ===
    // These method parameters are annotated, creating requirements for callers
    
    public void executeQuery(@SqlEvenQuotes String sql) {
        System.out.println("Executing: " + sql);
    }
    
    public void executeUpdate(@SqlEvenQuotes String sql) {
        System.out.println("Updating: " + sql);
    }
    
    public void prepareStatement(@SqlEvenQuotes String sql) {
        System.out.println("Preparing: " + sql);
    }
    
    // === WARNING SITES (unannotated) ===
    // These internal variables are NOT annotated.
    // Passing them to annotated methods creates warnings.
    // The model should learn to add @SqlEvenQuotes to fix warnings.
    
    // WARNING 1: Unannotated field passed to annotated method
    @SqlEvenQuotes
    private String query1 = "SELECT * FROM users";
    
    public void example1() {
        // This line creates a warning: query1 is unannotated
        executeQuery(query1);  // WARNING: [argument] incompatible
        // FIX: Add @SqlEvenQuotes to query1 field
    }
    
    // WARNING 2: Unannotated local variable
    public void example2() {
        String query2 = "SELECT * FROM orders WHERE id = ?";
        executeQuery(query2);  // WARNING: [argument] incompatible
        // FIX: Add @SqlEvenQuotes to query2
    }
    
    // WARNING 3: Unannotated method return value
    public String buildQuery() {
        return "UPDATE products SET price = ?";
    }
    
    public void example3() {
        executeUpdate(buildQuery());  // WARNING: [argument] incompatible
        // FIX: Add @SqlEvenQuotes to buildQuery() return type
    }
    
    // WARNING 4: String concatenation
    public void example4() {
        String table = "customers";
        String query4 = "SELECT * FROM " + table;
        executeQuery(query4);  // WARNING: [argument] incompatible
        // FIX: Add @SqlEvenQuotes to query4
    }
    
    // === CORRECT EXAMPLES (no warnings) ===
    // These show the pattern after fix is applied
    
    @SqlEvenQuotes String correctQuery = "SELECT * FROM users";
    
    public void correctExample() {
        executeQuery(correctQuery);  // No warning
    }
    
    public @SqlEvenQuotes String correctBuildQuery() {
        return "SELECT 1";
    }
    
    public void correctExample2() {
        executeQuery(correctBuildQuery());  // No warning
    }
}
