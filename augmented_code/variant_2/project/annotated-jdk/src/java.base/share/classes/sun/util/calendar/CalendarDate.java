/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 2000, 2011, Oracle and/or its affiliates. All rights reserved.
    @Positive
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @Positive
 * This code is free software; you can redistribute it and/or modify it
    @Positive
 * under the terms of the GNU General Public License version 2 only, as
    @Positive
 * published by the Free Software Foundation.  Oracle designates this
    @Positive
 * particular file as subject to the "Classpath" exception as provided
    @Positive
 * by Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @Positive
 * This code is distributed in the hope that it will be useful, but WITHOUT
    @Positive
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @Positive
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @Positive
 * version 2 for more details (a copy is included in the LICENSE file that
    @Positive
 * accompanied this code).
    @Positive
 *
    @Positive
 * You should have received a copy of the GNU General Public License version
    @Positive
 * 2 along with this work; if not, write to the Free Software Foundation,
    @Positive
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Positive
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @Positive
 * or visit www.oracle.com if you need additional information or have any
    @Positive
 * questions.
    @Positive
 */
    @Positive
package sun.util.calendar;

    @Positive
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
    @Positive
import org.checkerframework.checker.nullness.qual.NonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import java.lang.Cloneable;
    @Positive
import java.util.Locale;
    @Positive
import java.util.TimeZone;

    @Positive
public abstract class CalendarDate implements Cloneable {

    @Positive
    public static final int FIELD_UNDEFINED;

    @Positive
    public static final long TIME_UNDEFINED;

    @Positive
    protected CalendarDate() {
    @Positive
    }

    @Positive
    protected CalendarDate(TimeZone zone) {
    @Positive
    }

    @Positive
    public Era getEra();

    @Positive
    public CalendarDate setEra(Era era);

    @Positive
    public int getYear();

    @Positive
    public CalendarDate setYear(int year);

    @Positive
    public CalendarDate addYear(int n);

    @Positive
    public boolean isLeapYear();

    @Positive
    void setLeapYear(boolean leapYear);

    @Positive
    public int getMonth();

    @Positive
    public CalendarDate setMonth(int month);

    @Positive
    public CalendarDate addMonth(int n);

    @Positive
    public int getDayOfMonth();

    @Positive
    public CalendarDate setDayOfMonth(int date);

    @Positive
    public CalendarDate addDayOfMonth(int n);

    @Positive
    public int getDayOfWeek();

    @Positive
    public int getHours();

    @Positive
    public CalendarDate setHours(int hours);

    @Positive
    public CalendarDate addHours(int n);

    @Positive
    public int getMinutes();

    @Positive
    public CalendarDate setMinutes(int minutes);

    @Positive
    public CalendarDate addMinutes(int n);

    @Positive
    public int getSeconds();

    @Positive
    public CalendarDate setSeconds(int seconds);

    @Positive
    public CalendarDate addSeconds(int n);

    @Positive
    public int getMillis();

    @Positive
    public CalendarDate setMillis(int millis);

    @Positive
    public CalendarDate addMillis(int n);

    @Positive
    public long getTimeOfDay();

    @Positive
    public CalendarDate setDate(int year, int month, int dayOfMonth);

    @Positive
    public CalendarDate addDate(int year, int month, int dayOfMonth);

    @Positive
    public CalendarDate setTimeOfDay(int hours, int minutes, int seconds, int millis);

    @Positive
    public CalendarDate addTimeOfDay(int hours, int minutes, int seconds, int millis);

    @Positive
    protected void setTimeOfDay(long fraction);

    @Positive
    public boolean isNormalized();

    @Positive
    public boolean isStandardTime();

    @Positive
    public void setStandardTime(boolean standardTime);

    @Positive
    public boolean isDaylightTime();

    @Positive
    protected void setLocale(Locale loc);

    @Positive
    public TimeZone getZone();

    @Positive
    public CalendarDate setZone(TimeZone zoneinfo);

    @Positive
    public boolean isSameDate(CalendarDate date);

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    public int hashCode();

    @Positive
    public Object clone();

    @Positive
    public String toString();

    @Positive
    protected void setDayOfWeek(int dayOfWeek);

    @Positive
    protected void setNormalized(boolean normalized);

    @Positive
    public int getZoneOffset();

    @Positive
    protected void setZoneOffset(int offset);

    @Positive
    public int getDaylightSaving();

    @Positive
    protected void setDaylightSaving(int daylightSaving);
    @Positive
}
