package org.fr4j.labelProvider;

import org.datavec.image.recordreader.objdetect.ImageObject;
import org.datavec.image.recordreader.objdetect.ImageObjectLabelProvider;

import java.io.*;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

public final class FaceLabelProvider implements ImageObjectLabelProvider {

    private Map<String, List<ImageObject>> labelMap;

    public FaceLabelProvider(File dir) throws IOException {
        labelMap = new HashMap<>();

        ArrayList<String> Classes = new ArrayList<>(Files.readAllLines(Paths.get(dir.getPath() + "/_classes.txt")));


        for (String line : Files.readAllLines(Paths.get(dir.getPath() + "/_annotations.txt"))) {
            ArrayList<ImageObject> imageObjects = new ArrayList<>();

            String fileName = line.split(" ", 2)[0];

            String annotations = line.split(" ", 2)[1];

            for (String s : annotations.split(" ")){
                imageObjects.add(new ImageObject(
                        Integer.parseInt(s.split(",")[0]),
                        Integer.parseInt(s.split(",")[1]),
                        Integer.parseInt(s.split(",")[2]),
                        Integer.parseInt(s.split(",")[3]),
                        Classes.get(Integer.parseInt(s.split(",")[4]))
                ));
            }

            labelMap.put(fileName, imageObjects);
        }
    }

    @Override
    public List<ImageObject> getImageObjectsForPath(String path) {
        File file = new File(path);
        String filename = file.getName();

        return labelMap.get(filename);
    }

    @Override
    public List<ImageObject> getImageObjectsForPath(URI uri) {
        return getImageObjectsForPath(uri.toString());
    }
}
