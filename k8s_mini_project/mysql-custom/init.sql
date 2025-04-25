CREATE DATABASE IF NOT EXISTS mydb;

USE mydb;

CREATE TABLE IF NOT EXISTS employees (
    id INT AUTO_INCREMENT PRIMARY KEY,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    email VARCHAR(100),
    hire_date DATE
);

INSERT INTO employees (first_name, last_name, email, hire_date) VALUES
('John', 'Doe', 'john.doe@example.com', '2023-06-01'),
('Jane', 'Smith', 'jane.smith@example.com', '2023-07-15'),
('Sam', 'Green', 'sam.green@example.com', '2024-01-10'),
('Lily', 'Brown', 'lily.brown@example.com', '2024-03-22');