package com.example.beproject;

public class Prescription {
    public String date;
    public String imageUri;

    public Prescription(String date, String imageUri) {
        this.date = date;
        this.imageUri = imageUri;
    }

    public Prescription(String date) {
        this.date = date;
    }

    public Prescription() {
    }

    public String getDate() {

        return date;
    }

    public void setDate(String date) {
        this.date = date;
    }

    public String getImageUri() {
        return imageUri;
    }

    public void setImageUri(String imageUri) {
        this.imageUri = imageUri;
    }
}
