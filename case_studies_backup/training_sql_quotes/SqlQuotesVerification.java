// SQL Quotes Verification - Shows warning reduction by adding annotations
// Run this file with and without the @SqlEvenQuotes annotations to verify

import org.checkerframework.checker.sqlquotes.qual.SqlEvenQuotes;

public class SqlQuotesVerification {
    
    // Entry point (annotated)
    public void executeQuery(@SqlEvenQuotes String sql) {
        System.out.println(sql);
    }
    
    // BEFORE: Unannotated - creates warning
    // String query1 = "SELECT * FROM users";
    
    // AFTER: Annotated - no warning
    @SqlEvenQuotes String query1 = "SELECT * FROM users";
    
    // BEFORE: Unannotated return type - creates warning
    // public String buildQuery() { return "SELECT 1"; }
    
    // AFTER: Annotated return type - no warning  
    public @SqlEvenQuotes String buildQuery() { 
        return "SELECT 1"; 
    }
    
    public void test() {
        executeQuery(query1);     // No warning after adding @SqlEvenQuotes
        executeQuery(buildQuery()); // No warning after adding @SqlEvenQuotes
    }
}
