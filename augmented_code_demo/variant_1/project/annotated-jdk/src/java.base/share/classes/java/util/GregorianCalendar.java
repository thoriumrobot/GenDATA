/*
    @Positive
 * Copyright (c) 1996, 2020, Oracle and/or its affiliates. All rights reserved.
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
package java.util;

    @Positive
import org.checkerframework.checker.lock.qual.GuardSatisfied;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.common.value.qual.IntVal;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import org.checkerframework.framework.qual.CFComment;
    @Positive
import java.io.IOException;
    @Positive
import java.io.ObjectInputStream;
    @Positive
import java.time.Instant;
    @Positive
import java.time.ZonedDateTime;
    @Positive
import java.time.temporal.ChronoField;
    @Positive
import sun.util.calendar.BaseCalendar;
    @Positive
import sun.util.calendar.CalendarDate;
    @Positive
import sun.util.calendar.CalendarSystem;
    @Positive
import sun.util.calendar.CalendarUtils;
    @Positive
import sun.util.calendar.Era;
    @Positive
import sun.util.calendar.Gregorian;
    @Positive
import sun.util.calendar.JulianCalendar;
    @Positive
import sun.util.calendar.ZoneInfo;

    @Positive
@AnnotatedFor({ "lock", "nullness", "index" })
    @Positive
public class GregorianCalendar extends Calendar {

    @Positive
    @IntVal({ 0 })
    @Positive
    public static final int BC;

    @Positive
    @IntVal({ 1 })
    @Positive
    public static final int AD;

    @Positive
    public GregorianCalendar() {
    @Positive
    }

    @Positive
    public GregorianCalendar(TimeZone zone) {
    @Positive
    }

    @Positive
    public GregorianCalendar(Locale aLocale) {
    @Positive
    }

    @Positive
    public GregorianCalendar(TimeZone zone, Locale aLocale) {
    @Positive
    }

    @Positive
    public GregorianCalendar(int year, int month, int dayOfMonth) {
    @Positive
    }

    @Positive
    public GregorianCalendar(int year, int month, int dayOfMonth, int hourOfDay, int minute) {
    @Positive
    }

    @Positive
    public GregorianCalendar(int year, int month, int dayOfMonth, int hourOfDay, int minute, int second) {
    @Positive
    }

    @Positive
    public void setGregorianChange(@GuardSatisfied GregorianCalendar this, Date date);

    @Positive
    public final Date getGregorianChange();

    @Positive
    @Pure
    @Positive
    public boolean isLeapYear(@GuardSatisfied GregorianCalendar this, int year);

    @Positive
    @Override
    @Positive
    public String getCalendarType();

    @Positive
    @Pure
    @Positive
    @Override
    @Positive
    public boolean equals(@GuardSatisfied GregorianCalendar this, @GuardSatisfied @Nullable Object obj);

    @Positive
    @Pure
    @Positive
    @Override
    @Positive
    public int hashCode(@GuardSatisfied GregorianCalendar this);

    @Positive
    @Override
    @Positive
    public void add(@GuardSatisfied GregorianCalendar this, int field, int amount);

    @Positive
    @Override
    @Positive
    public void roll(@GuardSatisfied GregorianCalendar this, int field, boolean up);

    @Positive
    @Override
    @Positive
    public void roll(@GuardSatisfied GregorianCalendar this, int field, int amount);

    @Positive
    @Override
    @Positive
    public int getMinimum(int field);

    @Positive
    @Override
    @Positive
    public int getMaximum(int field);

    @Positive
    @Override
    @Positive
    public int getGreatestMinimum(int field);

    @Positive
    @Override
    @Positive
    public int getLeastMaximum(int field);

    @Positive
    @Override
    @Positive
    public int getActualMinimum(int field);

    @Positive
    @Override
    @Positive
    public int getActualMaximum(int field);

    @Positive
    @SideEffectFree
    @Positive
    @Override
    @Positive
    public Object clone(@GuardSatisfied GregorianCalendar this);

    @Positive
    @Override
    @Positive
    public TimeZone getTimeZone();

    @Positive
    @Override
    @Positive
    public void setTimeZone(@GuardSatisfied GregorianCalendar this, TimeZone zone);

    @Positive
    @Override
    @Positive
    public final boolean isWeekDateSupported();

    @Positive
    @Override
    @Positive
    public int getWeekYear();

    @Positive
    @Override
    @Positive
    public void setWeekDate(int weekYear, int weekOfYear, int dayOfWeek);

    @Positive
    @Override
    @Positive
    public int getWeeksInWeekYear();

    @Positive
    @Override
    @Positive
    protected void computeFields();

    @Positive
    @Override
    @Positive
    protected void computeTime();

    @Positive
    public ZonedDateTime toZonedDateTime();

    @Positive
    public static GregorianCalendar from(ZonedDateTime zdt);
    @Positive
}

// CFWR semantic augmentation - variant 1
