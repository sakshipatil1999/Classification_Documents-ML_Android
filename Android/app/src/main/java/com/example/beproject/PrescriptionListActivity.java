package com.example.beproject;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import android.content.Intent;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Toolbar;

import com.firebase.ui.database.FirebaseRecyclerAdapter;
import com.firebase.ui.database.FirebaseRecyclerOptions;
import com.firebase.ui.database.SnapshotParser;
import com.google.firebase.auth.FirebaseAuth;
import com.google.firebase.database.ChildEventListener;
import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;

import java.util.ArrayList;
import java.util.List;

public class PrescriptionListActivity extends AppCompatActivity {
    private RecyclerView mPrescriptionList;
    private DatabaseReference mDatabase;
    private androidx.appcompat.widget.Toolbar mToolbar;
    private FirebaseAuth mAuth;
    private final List<Prescription> prescriptionList=new ArrayList<>();
    private LinearLayoutManager mLinearLayout;
    private PrescriptionAdapter mAdapter;
    private String mCurrentUserId;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_prescription_list);

        mToolbar= findViewById(R.id.all_user_appbar);
        setSupportActionBar(mToolbar);
        getSupportActionBar().setTitle("Prescriptions");
        getSupportActionBar().setDisplayShowHomeEnabled(true);
        getSupportActionBar().setDisplayHomeAsUpEnabled(true);

        mToolbar.setNavigationOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                startActivity (new Intent(PrescriptionListActivity.this,MainActivity.class));
            }
        });

        mPrescriptionList= findViewById(R.id.prescription_list);
        
        mAdapter=new PrescriptionAdapter(prescriptionList);
        mLinearLayout=new LinearLayoutManager(this);


        mPrescriptionList.setHasFixedSize(true);
        mPrescriptionList.setLayoutManager(mLinearLayout);
        mPrescriptionList.setAdapter(mAdapter);

        mAuth=FirebaseAuth.getInstance();
        //Get current user id
        mCurrentUserId=mAuth.getCurrentUser().getUid();
        mDatabase=FirebaseDatabase.getInstance().getReference();
        loadPrescription();

    }



    private void loadPrescription(){

        //loading of message on screen

        mDatabase.child("Users").child(mCurrentUserId).child("Prescription").addChildEventListener(new ChildEventListener() {
            @Override
            public void onChildAdded(@NonNull DataSnapshot dataSnapshot, @Nullable String s) {

                Prescription message=dataSnapshot.getValue(Prescription.class);
                prescriptionList.add(message);
                mAdapter.notifyDataSetChanged();

            }

            @Override
            public void onChildChanged(@NonNull DataSnapshot dataSnapshot, @Nullable String s) {

            }

            @Override
            public void onChildRemoved(@NonNull DataSnapshot dataSnapshot) {
                Prescription message=dataSnapshot.getValue(Prescription.class);
                prescriptionList.add(message);
                mAdapter.notifyDataSetChanged();

            }

            @Override
            public void onChildMoved(@NonNull DataSnapshot dataSnapshot, @Nullable String s) {

            }

            @Override
            public void onCancelled(@NonNull DatabaseError databaseError) {

            }
        });



    }



}