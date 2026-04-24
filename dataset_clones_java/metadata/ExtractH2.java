import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.Statement;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.sql.ResultSetMetaData;

public class ExtractH2 {
    public static void main(String[] args) {
        String jdbcUrl = "jdbc:h2:./bcb;AUTO_SERVER=TRUE";
        String user = "sa";
        String password = "";

        try (Connection conn = DriverManager.getConnection(jdbcUrl, user, password);
             Statement stmt = conn.createStatement();
             ResultSet rs = stmt.executeQuery("SELECT FUNCTION_ID_ONE, FUNCTION_ID_TWO, SYNTACTIC_TYPE, SIMILARITY_LINE, SIMILARITY_TOKEN FROM CLONES");
             PrintWriter pw = new PrintWriter(new FileWriter("clones.csv"))) {
             
            ResultSetMetaData rsmd = rs.getMetaData();
            int columnCount = rsmd.getColumnCount();
            
            // Print headers
            for (int i = 1; i <= columnCount; i++) {
                pw.print(rsmd.getColumnName(i));
                if (i < columnCount) pw.print(",");
            }
            pw.println();
            
            // Print rows
            while (rs.next()) {
                for (int i = 1; i <= columnCount; i++) {
                    pw.print(rs.getString(i));
                    if (i < columnCount) pw.print(",");
                }
                pw.println();
            }
            System.out.println("Extraction successful! clones.csv generated.");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
