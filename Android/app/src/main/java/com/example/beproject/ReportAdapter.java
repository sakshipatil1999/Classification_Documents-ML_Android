package com.example.beproject;

import android.content.Intent;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import com.google.firebase.auth.FirebaseAuth;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;

import org.jetbrains.annotations.NotNull;

import java.util.List;

public class ReportAdapter extends RecyclerView.Adapter<ReportAdapter.ReportViewHolder> {
    private FirebaseAuth mAuth;
    private List<Report> mMessageList;
    private DatabaseReference mUserDatabase;

    public ReportAdapter(List<Report> mMessageList) {
        this.mMessageList = mMessageList;
    }


    @Override
    public ReportViewHolder onCreateViewHolder(ViewGroup parent, int viewType){

        mAuth=FirebaseAuth.getInstance();
        View v= LayoutInflater.from(parent.getContext()).inflate(R.layout.list,parent,false);
        return new ReportViewHolder(v);
    }

    @Override
    public void onBindViewHolder(@NonNull @NotNull ReportViewHolder holder, int position) {
        String current_user_id=mAuth.getCurrentUser().getUid();
        mUserDatabase= FirebaseDatabase.getInstance().getReference().child("Users");
        mUserDatabase.keepSynced(true);

        final Report c=mMessageList.get(position);

        String date=c.getDate();
        String imageUri=c.getImageUri();

        holder.date.setText(date);
        holder.mView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent displayImg = new Intent(view.getContext(),ImageDisplayActivity.class);
                displayImg.putExtra("imageUri",imageUri);
                view.getContext().startActivity(displayImg);
            }
        });


    }

    @Override
    public int getItemCount() {
        return mMessageList.size();
    }

    public class ReportViewHolder extends RecyclerView.ViewHolder{
        public TextView date;
        public View mView;

        public ReportViewHolder(@NonNull View itemView) {
            super(itemView);

            date=itemView.findViewById(R.id.date);
            mView = itemView.findViewById(R.id.list);

        }

    }
}
