// Test fixture for SQL Quotes annotation placement
// This file contains various SQL patterns for testing

public class SqlQuotesSample {
    
    // Simple SQL query - should get @SqlEvenQuotes (no single quotes)
    String simpleQuery = "SELECT * FROM users";
    
    // SQL with even quotes (2 quotes = valid)
    String queryWithQuotes = "SELECT * FROM users WHERE name = ''";
    
    // SQL with parameters
    String parameterizedQuery = "INSERT INTO users (name, age) VALUES (?, ?)";
    
    // Multi-line SQL
    String multiLineQuery = "SELECT u.id, u.name " +
                           "FROM users u " +
                           "WHERE u.active = true";
    
    // Method with SQL parameter
    public void executeQuery(String sql) {
        System.out.println("Executing: " + sql);
    }
    
    // Method returning SQL
    public String buildQuery() {
        return "DELETE FROM users WHERE id = ?";
    }
    
    // PreparedStatement usage
    public void prepareStatement(String sql) {
        // Database operation
    }
    
    // SQL UPDATE statement
    public void runUpdate() {
        String updateSql = "UPDATE users SET active = false WHERE last_login < ?";
        executeQuery(updateSql);
    }
    
    // JOIN query
    public void complexQuery() {
        String joinQuery = "SELECT u.name, o.total FROM users u JOIN orders o ON u.id = o.user_id";
        executeQuery(joinQuery);
    }
    
    // Non-SQL string (should NOT be annotated)
    String regularString = "This is not SQL";
    
    // Field declaration only
    String pendingQuery;
}
