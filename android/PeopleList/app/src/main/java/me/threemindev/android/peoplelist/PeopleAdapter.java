package me.threemindev.android.peoplelist;

import android.content.Context;
import android.provider.Contacts;
import android.support.annotation.NonNull;
import android.support.annotation.Nullable;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import android.widget.ListView;
import android.widget.TextView;

import java.util.ArrayList;

/**
 * Created by newtong on 2017-04-04.
 */

public class PeopleAdapter extends ArrayAdapter<Person> {
    public PeopleAdapter(Context context, ArrayList<Person> peopleList) {
        super(context,0,peopleList);
    }

    @NonNull
    @Override
    public View getView(int position, View convertView, ViewGroup parent) {
        if(convertView == null) {
            LayoutInflater layoutInflator = LayoutInflater.from(getContext());
            convertView = layoutInflator.inflate(R.layout.listviewitemperson, parent,false);
        }

        TextView nameTextView = (TextView)convertView.findViewById(R.id.nameTextView);
        nameTextView.setText(getItem(position).name);

        return convertView;
    }
}
