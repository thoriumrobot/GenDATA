/*
 * Copyright (c) 2000, 2011, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */
package sun.util.calendar;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.lang.Cloneable;
import java.util.Locale;
import java.util.TimeZone;

public abstract class CalendarDate implements Cloneable {

    public static final int FIELD_UNDEFINED;

    public static final long TIME_UNDEFINED;

    protected CalendarDate() {
    }

    protected CalendarDate(TimeZone zone) {
    }

    public Era getEra();

    public CalendarDate setEra(Era era);

    public int getYear();

    public CalendarDate setYear(int year);

    public CalendarDate addYear(int n);

    public boolean isLeapYear();

    void setLeapYear(boolean leapYear);

    public int getMonth();

    public CalendarDate setMonth(int month);

    public CalendarDate addMonth(int n);

    public int getDayOfMonth();

    public CalendarDate setDayOfMonth(int date);

    public CalendarDate addDayOfMonth(int n);

    public int getDayOfWeek();

    public int getHours();

    public CalendarDate setHours(int hours);

    public CalendarDate addHours(int n);

    public int getMinutes();

    public CalendarDate setMinutes(int minutes);

    public CalendarDate addMinutes(int n);

    public int getSeconds();

    public CalendarDate setSeconds(int seconds);

    public CalendarDate addSeconds(int n);

    public int getMillis();

    public CalendarDate setMillis(int millis);

    public CalendarDate addMillis(int n);

    public long getTimeOfDay();

    public CalendarDate setDate(int year, int month, int dayOfMonth);

    public CalendarDate addDate(int year, int month, int dayOfMonth);

    public CalendarDate setTimeOfDay(int hours, int minutes, int seconds, int millis);

    public CalendarDate addTimeOfDay(int hours, int minutes, int seconds, int millis);

    protected void setTimeOfDay(long fraction);

    public boolean isNormalized();

    public boolean isStandardTime();

    public void setStandardTime(boolean standardTime);

    public boolean isDaylightTime();

    protected void setLocale(Locale loc);

    public TimeZone getZone();

    public CalendarDate setZone(TimeZone zoneinfo);

    public boolean isSameDate(CalendarDate date);

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    public int hashCode();

    public Object clone();

    public String toString();

    protected void setDayOfWeek(int dayOfWeek);

    protected void setNormalized(boolean normalized);

    public int getZoneOffset();

    protected void setZoneOffset(int offset);

    public int getDaylightSaving();

    protected void setDaylightSaving(int daylightSaving);
}
