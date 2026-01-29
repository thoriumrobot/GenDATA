// SQL Quotes - BEFORE adding annotations (has warnings)
// This file shows the state BEFORE the model adds annotations
// Running the checker on this file produces warnings

import org.checkerframework.checker.sqlquotes.qual.SqlOddQuotes;
import org.checkerframework.checker.sqlquotes.qual.SqlEvenQuotes;

public class SqlQuotesBeforeAnnotation {
    
    // Entry point (annotated) - this stays annotated
    public void executeQuery(@SqlEvenQuotes String sql) {
        System.out.println(sql);
    }
    
    // UNANNOTATED - will cause warning
    @SqlEvenQuotes
    String query1 = "SELECT * FROM users";
    
    // UNANNOTATED return type - will cause warning
    public String buildQuery() { 
        return "SELECT 1"; 
    }
    
    public void test() {
        executeQuery(query1);       // WARNING: incompatible argument
        executeQuery(buildQuery()); // WARNING: incompatible argument
    }
}
