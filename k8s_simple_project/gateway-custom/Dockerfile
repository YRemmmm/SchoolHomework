FROM openjdk:17-jdk-alpine
ADD Gateway-0.0.1-SNAPSHOT.jar /gateway.jar
CMD java -jar /gateway.jar --spring.profiles.active=test
EXPOSE 8080
