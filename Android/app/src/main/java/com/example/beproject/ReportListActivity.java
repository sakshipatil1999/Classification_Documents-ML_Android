package com.example.beproject;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;

import com.google.firebase.auth.FirebaseAuth;
import com.google.firebase.database.ChildEventListener;
import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;

import java.util.ArrayList;
import java.util.List;

public class ReportListActivity extends AppCompatActivity {
    private RecyclerView mReportList;

    private DatabaseReference mDatabase;
    private androidx.appcompat.widget.Toolbar mToolbar;
    private FirebaseAuth mAuth;
    private final List<Report> reportList =new ArrayList<>();
    private LinearLayoutManager mLinearLayout;
    private ReportAdapter mAdapter;
    private String mCurrentUserId;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_report_list);

        mToolbar= findViewById(R.id.all_user_appbar);
        setSupportActionBar(mToolbar);
        getSupportActionBar().setTitle("Reports");
        getSupportActionBar().setDisplayShowHomeEnabled(true);
        getSupportActionBar().setDisplayHomeAsUpEnabled(true);

        mToolbar.setNavigationOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                startActivity (new Intent(ReportListActivity.this,MainActivity.class));
            }
        });

        mReportList= findViewById(R.id.report_list);

        mAdapter=new ReportAdapter(reportList);
        mLinearLayout=new LinearLayoutManager(this);


        mReportList.setHasFixedSize(true);
        mReportList.setLayoutManager(mLinearLayout);
        mReportList.setAdapter(mAdapter);

        mAuth=FirebaseAuth.getInstance();
        //Get current user id
        mCurrentUserId=mAuth.getCurrentUser().getUid();
        mDatabase= FirebaseDatabase.getInstance().getReference();
        loadReport();

    }

    private void loadReport(){

        //loading of message on screen

        mDatabase.child("Users").child(mCurrentUserId).child("Report").addChildEventListener(new ChildEventListener() {
            @Override
            public void onChildAdded(@NonNull DataSnapshot dataSnapshot, @Nullable String s) {

                Report message=dataSnapshot.getValue(Report.class);
                reportList.add(message);
                mAdapter.notifyDataSetChanged();

            }

            @Override
            public void onChildChanged(@NonNull DataSnapshot dataSnapshot, @Nullable String s) {

            }

            @Override
            public void onChildRemoved(@NonNull DataSnapshot dataSnapshot) {
                Report message=dataSnapshot.getValue(Report.class);
                reportList.add(message);
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