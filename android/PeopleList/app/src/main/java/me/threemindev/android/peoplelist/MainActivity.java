package me.threemindev.android.peoplelist;

import android.support.annotation.Nullable;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.widget.ArrayAdapter;
import android.widget.ListView;

import java.lang.reflect.Array;
import java.util.ArrayList;

public class MainActivity extends AppCompatActivity {

    ListView peopleListView;
    PeopleAdapter peopleAdapter;
    ArrayList<Person> peopleList;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        {
            //Model ( Actual Data )
            peopleList = new ArrayList<>();
            peopleList.add(new Person());

            //Controller
            peopleAdapter = new PeopleAdapter((MainActivity)this, (ArrayList<Person>)peopleList);

            //View
            peopleListView = (ListView) findViewById(R.id.peopleListView);
            peopleListView.setAdapter(peopleAdapter);
        }
    }
}
