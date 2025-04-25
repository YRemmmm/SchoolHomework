<%@ page import="java.sql.*" %>
<%@ page contentType="text/html;charset=UTF-8" language="java" %>
<%
    /////////////////////////////////////////////////////////////////////////////////
    String host = "10.233.0.10:3306";	//数据库地址和端口
    String password = "123456";	//数据库密码
    String dbName = "mydb";	//数据库名
    String table = "employees";		//数据库表名
    /////////////////////////////////////////////////////////////////////////////////
    String url = "jdbc:mysql://" + host + "/" + dbName + "?useSSL=false&useUnicode=true&characterEncoding=UTF-8";
    Class.forName("com.mysql.jdbc.Driver");
    Connection connection = DriverManager.getConnection(url, "root", password);
%>
<html>
<head>
    <title><%=dbName%>.<%=table%></title>
    <style>
        table{
            border-collapse: collapse;
        }
        table td, table th{
            border: 1px #000000 solid;
            padding: 5px;
        }
        table th{
            background-color: aliceblue;
            padding: 5px;
        }
    </style>
</head>
<body>
<h1>This is Cluster 1 Data Display Page</h1>
<%
    String sql = "select * from " + table;
    Statement statement = connection.createStatement();
    ResultSet rs = statement.executeQuery(sql);
    ResultSetMetaData cols = rs.getMetaData();
    int num = cols.getColumnCount();
    String html = "<table><tr>";
    for (int i = 1; i <= num; i ++) {
        html += "<th>" + cols.getColumnName(i) + "</th>";
    }
    html += "</tr>";
    while(rs.next()) {
        html+= "<tr>";
        for (int i = 1; i <= num; i ++) {
            html += "<td>" + rs.getObject(cols.getColumnName(i)) + "</td>";
        }
        html+= "</tr>";
    }
    html += "</table>";
    rs.close();
    statement.close();
    connection.close();
    response.getWriter().println(host + "<br>");
    response.getWriter().println(dbName + "." + table + "<br>");
    response.getWriter().println(html);
%>
</body>
</html>
