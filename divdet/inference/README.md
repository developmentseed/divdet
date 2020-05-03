# Postgres Trigger
Below is a database trigger function that can be run prior to any database inserts. It'll look for geometric overlaps and 
1. If a crater overlap is found with lower confidence, delete the existing crater and insert the new one
1. If a crater overlap is found with higher confidence, abort the crater insert

```
CREATE OR REPLACE FUNCTION crater_dedup_insert() RETURNS TRIGGER AS $crater_dedup_insert$
    DECLARE
        crater_id int;
    BEGIN
        -- Remove any existing, intersecting craters with a lower confidence
        DELETE FROM craters WHERE st_intersects(craters.geom, NEW.geom) and NEW.confidence > craters.confidence; 

        -- Skip crater if it intersects with any existing (which we now know are higher confidence)
        SELECT 1 FROM craters WHERE st_intersects (craters.geom, NEW.geom);
        IF  FOUND THEN
            RETURN NULL;

        END IF;
        -- Add new crater that doesn't intersect any craters of higher confidence
        RETURN NEW;
    END;
$crater_dedup_insert$ LANGUAGE plpgsql;
```

Apply the trigger to the table like:
```
CREATE TRIGGER crater_dedup_insert 
BEFORE INSERT ON craters
    FOR EACH ROW EXECUTE PROCEDURE crater_dedup_insert();
```

# Indexing the table
We also likely want to create an index on the table to speed things up:
```
CREATE INDEX ON craters USING GIST (geom, confidence);
```

# Checking IOU
ST_Area(ST_Intersection(NEW.geom, craters.geom)) / ST_Area(ST_Union(NEW.geom, craters.geom)) > craters.iou_threshold