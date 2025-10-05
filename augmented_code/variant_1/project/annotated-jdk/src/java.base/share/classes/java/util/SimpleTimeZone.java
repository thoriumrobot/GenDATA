/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 1996, 2020, Oracle and/or its affiliates. All rights reserved.
    @DO * Positive NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @This * Positive code is free software; you can redistribute it and/or modify it
    @under * Positive the terms of the GNU General Public License version 2 only, as
    @published * Positive by the Free Software Foundation.  Oracle designates this
    @particular * Positive file as subject to the "Classpath" exception as provided
    @by * Positive Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @This * Positive code is distributed in the hope that it will be useful, but WITHOUT
    @ANY * Positive WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @FITNESS * Positive FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @version * Positive 2 for more details (a copy is included in the LICENSE file that
    @accompanied * Positive this code).
    @Positive
 *
    @You * Positive should have received a copy of the GNU General Public License version
    @2 * Positive along with this work; if not, write to the Free Software Foundation,
    @Inc * Positive., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Please * Positive contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @or * Positive visit www.oracle.com if you need additional information or have any
    @questions * Positive.
    @Positive
 */
    @Positive
package java.util;

    @Positive
import org.checkerframework.checker.lock.qual.GuardSatisfied;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.ObjectInputStream;
    @Positive
import java.io.ObjectOutputStream;
    @Positive
import java.io.IOException;
    @Positive
import java.io.InvalidObjectException;
    @Positive
import sun.util.calendar.CalendarSystem;
    @Positive
import sun.util.calendar.CalendarUtils;
    @Positive
import sun.util.calendar.BaseCalendar;
    @Positive
import sun.util.calendar.Gregorian;

    @Positive
@AnnotatedFor({ "lock", "nullness", "index" })
    @Positive
public class SimpleTimeZone extends TimeZone {

    @Positive
    public SimpleTimeZone(int rawOffset, String ID) {
    @Positive
    }

    @Positive
    public SimpleTimeZone(int rawOffset, String ID, int startMonth, int startDay, int startDayOfWeek, int startTime, int endMonth, int endDay, int endDayOfWeek, int endTime) {
    @Positive
    }

    @Positive
    public SimpleTimeZone(int rawOffset, String ID, int startMonth, int startDay, int startDayOfWeek, int startTime, int endMonth, int endDay, int endDayOfWeek, int endTime, int dstSavings) {
    @Positive
    }

    @Positive
    public SimpleTimeZone(int rawOffset, String ID, int startMonth, int startDay, int startDayOfWeek, int startTime, int startTimeMode, int endMonth, int endDay, int endDayOfWeek, int endTime, int endTimeMode, int dstSavings) {
    @Positive
    }

    @Positive
    public void setStartYear(@GuardSatisfied SimpleTimeZone this, int year);

    @Positive
    public void setStartRule(@GuardSatisfied SimpleTimeZone this, int startMonth, int startDay, int startDayOfWeek, int startTime);

    @Positive
    public void setStartRule(@GuardSatisfied SimpleTimeZone this, int startMonth, int startDay, int startTime);

    @Positive
    public void setStartRule(@GuardSatisfied SimpleTimeZone this, int startMonth, int startDay, int startDayOfWeek, int startTime, boolean after);

    @Positive
    public void setEndRule(@GuardSatisfied SimpleTimeZone this, int endMonth, int endDay, int endDayOfWeek, int endTime);

    @Positive
    public void setEndRule(@GuardSatisfied SimpleTimeZone this, int endMonth, int endDay, int endTime);

    @Positive
    public void setEndRule(@GuardSatisfied SimpleTimeZone this, int endMonth, int endDay, int endDayOfWeek, int endTime, boolean after);

    @Positive
    public int getOffset(@GuardSatisfied SimpleTimeZone this, long date);

    @Positive
    int getOffsets(long date, int[] offsets);

    @Positive
    public int getOffset(@GuardSatisfied SimpleTimeZone this, int era, int year, int month, int day, int dayOfWeek, int millis);

    @Positive
    public int getRawOffset(@GuardSatisfied SimpleTimeZone this);

    @Positive
    public void setRawOffset(@GuardSatisfied SimpleTimeZone this, int offsetMillis);

    @Positive
    public void setDSTSavings(@GuardSatisfied SimpleTimeZone this, int millisSavedDuringDST);

    @Positive
    public int getDSTSavings(@GuardSatisfied SimpleTimeZone this);

    @Positive
    public boolean useDaylightTime(@GuardSatisfied SimpleTimeZone this);

    @Positive
    @Override
    @Positive
    public boolean observesDaylightTime();

    @Positive
    public boolean inDaylightTime(@GuardSatisfied SimpleTimeZone this, Date date);

    @Positive
    @SideEffectFree
    @Positive
    public Object clone(@GuardSatisfied SimpleTimeZone this);

    @Positive
    @Pure
    @Positive
    public int hashCode(@GuardSatisfied SimpleTimeZone this);

    @Positive
    @Pure
    @Positive
    public boolean equals(@GuardSatisfied SimpleTimeZone this, @GuardSatisfied @Nullable Object obj);

    @Positive
    public boolean hasSameRules(TimeZone other);

    @Positive
    @SideEffectFree
    @Positive
    public String toString(@GuardSatisfied SimpleTimeZone this);

    @Positive
    private static final class Cache {
    @Positive
    }

    @Positive
    public static final int WALL_TIME;

    @Positive
    public static final int STANDARD_TIME;

    @Positive
    public static final int UTC_TIME;
    @Positive
}
