package com.example.beproject;

import androidx.appcompat.app.ActionBarDrawerToggle;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;
import androidx.drawerlayout.widget.DrawerLayout;

import android.content.Intent;
import android.os.Bundle;
import android.view.MenuItem;
import android.view.View;
import android.widget.TextView;

import com.google.android.material.navigation.NavigationView;

public class HomeActivity extends AppCompatActivity {
    DrawerLayout drawerLayout;
    Toolbar toolbar;
    NavigationView navigationView;
    ActionBarDrawerToggle toggle;
    TextView tPrescription;
    TextView tReport;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_home);

        setContentView(R.layout.activity_home);
        drawerLayout=findViewById(R.id.drawer);
        toolbar= findViewById(R.id.toolbar);
        navigationView=findViewById(R.id.navigationView);
        setSupportActionBar(toolbar);
        getSupportActionBar().setDisplayHomeAsUpEnabled(true);
        getSupportActionBar().setDisplayHomeAsUpEnabled(true);
        navigationView.setNavigationItemSelectedListener(    new NavigationView.OnNavigationItemSelectedListener() {
            @Override
            public boolean onNavigationItemSelected(MenuItem menuItem) {
                // set item as selected to persist highlight
                //menuItem.setChecked(true);
                // close drawer when item is tapped
                switch(menuItem.getItemId()){
                    case R.id.imageactivity:
                        Intent upcoming=new Intent(getApplicationContext(), ImageActivity.class);
                        startActivity(upcoming);
                        break;

                    case R.id.logout:
                        Intent logout =new Intent(getApplicationContext(),LoginActivity.class);
                        startActivity(logout);
                        break;

                }
                drawerLayout.closeDrawers();
                // Add code here to update the UI based on the item selected
                // For example, swap UI fragments here
                return false;
            }

        });
        toggle=new ActionBarDrawerToggle(this,drawerLayout,toolbar,R.string.drawerOpen,R.string.drawerClose);
        drawerLayout.addDrawerListener(toggle);
        toggle.syncState();

        tPrescription =(TextView) findViewById(R.id.prescription);
        tReport= (TextView) findViewById(R.id.report);

        tPrescription.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent pres = new Intent(getBaseContext(),PrescriptionListActivity.class);
                startActivity(pres);

            }
        });

        tReport.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent report = new Intent(getBaseContext(),ReportListActivity.class);
                startActivity(report);

            }
        });

    }
}