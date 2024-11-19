package com.example.beproject;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.text.TextUtils;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Toast;

import com.google.android.gms.tasks.OnCompleteListener;
import com.google.android.gms.tasks.Task;
import com.google.firebase.auth.AuthResult;
import com.google.firebase.auth.FirebaseAuth;
import com.google.firebase.auth.FirebaseUser;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;

import java.util.regex.Pattern;

public class SignUpActivity extends AppCompatActivity {

    private FirebaseAuth mAuth;
    EditText email;
    EditText password;
    Button submit;
    private DatabaseReference mDatabase;
    private FirebaseDatabase database;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_sign_up);
        email =(EditText) findViewById(R.id.signupEmail);
        password=(EditText) findViewById(R.id.signupPassword);
        submit =(Button) findViewById(R.id.signup);

        // Initialize Firebase Auth
        mAuth = FirebaseAuth.getInstance();

        submit.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
            String sEmail=email.getText().toString();
            String sPwd =password.getText().toString();
                if (isValid(sEmail)) {

                    if (!TextUtils.isEmpty(sEmail) && !TextUtils.isEmpty(sPwd)) {

                        register(sEmail,sPwd);
                    }
                }else {

                    Toast.makeText(SignUpActivity.this, "Enter Correct Email", Toast.LENGTH_SHORT).show();
                }


            }
        });


    }

    @Override
    public void onStart() {
        super.onStart();
        // Check if user is signed in (non-null) and update UI accordingly.
        FirebaseUser currentUser = mAuth.getCurrentUser();

    }

    private void register(String email, String pwd){

        mAuth.createUserWithEmailAndPassword(email, pwd)
                .addOnCompleteListener(this, new OnCompleteListener<AuthResult>() {
                    @Override
                    public void onComplete(@NonNull Task<AuthResult> task) {
                        if (task.isSuccessful()) {
                            // Sign in success, update UI with the signed-in user's information
                            FirebaseUser user = mAuth.getCurrentUser();
                            String uid=user.getUid();
                            database= FirebaseDatabase.getInstance();
                            mDatabase=database.getReference().child("Users").child(uid);
                            Intent intent = new Intent(getBaseContext(), HomeActivity.class);
                            startActivity(intent);
                        } else {
                            // If sign in fails, display a message to the user.

                            Toast.makeText(SignUpActivity.this, "Authentication failed.",
                                    Toast.LENGTH_SHORT).show();

                        }

                        // ...
                    }
                });
    }

    public static boolean isValid(String email)
    {
        String emailRegex = "^[a-zA-Z0-9_+&*-]+(?:\\."+
                "[a-zA-Z0-9_+&*-]+)*@" +
                "(?:[a-zA-Z0-9-]+\\.)"+"(com|in|org)"+
                "$";

        Pattern pat = Pattern.compile(emailRegex);
        if (email == null)
            return false;
        return pat.matcher(email).matches();
    }


}