package com.example.beproject;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;

import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;
import com.google.firebase.auth.FirebaseAuth;
import com.google.firebase.auth.FirebaseUser;

public class MainActivity extends AppCompatActivity {

    private FirebaseAuth mAuth;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        mAuth = FirebaseAuth.getInstance();

        if (!Python.isStarted()) {
            Python.start(new AndroidPlatform(getBaseContext()));
        }

        Intent intent = new Intent(getBaseContext(),SignUpActivity.class);
        startActivity(intent);

    }

    @Override
    public void onStart() {
        super.onStart();
        // Check if user is signed in (non-null) and update UI accordingly.
        FirebaseUser currentUser = mAuth.getCurrentUser();

        if(currentUser==null){
            sendToLogin();
        }else{
            sendToHome();
        }
    }
    public void sendToLogin() {
        Intent logIn = new Intent(getBaseContext(), LoginActivity.class);
        startActivity(logIn);
        finish();
    }
    public void sendToHome(){
        Intent home = new Intent(getBaseContext(), HomeActivity.class);
        startActivity(home);
        finish();
    }

}