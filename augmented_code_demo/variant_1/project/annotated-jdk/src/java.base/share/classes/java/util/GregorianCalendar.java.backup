/*
 * Copyright (c) 1996, 2020, Oracle and/or its affiliates. All rights reserved.
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
package java.util;

import org.checkerframework.checker.lock.qual.GuardSatisfied;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.common.value.qual.IntVal;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import org.checkerframework.framework.qual.AnnotatedFor;
import org.checkerframework.framework.qual.CFComment;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.time.Instant;
import java.time.ZonedDateTime;
import java.time.temporal.ChronoField;
import sun.util.calendar.BaseCalendar;
import sun.util.calendar.CalendarDate;
import sun.util.calendar.CalendarSystem;
import sun.util.calendar.CalendarUtils;
import sun.util.calendar.Era;
import sun.util.calendar.Gregorian;
import sun.util.calendar.JulianCalendar;
import sun.util.calendar.ZoneInfo;

@AnnotatedFor({ "lock", "nullness", "index" })
public class GregorianCalendar extends Calendar {

    @IntVal({ 0 })
    public static final int BC;

    @IntVal({ 1 })
    public static final int AD;

    public GregorianCalendar() {
    }

    public GregorianCalendar(TimeZone zone) {
    }

    public GregorianCalendar(Locale aLocale) {
    }

    public GregorianCalendar(TimeZone zone, Locale aLocale) {
    }

    public GregorianCalendar(int year, int month, int dayOfMonth) {
    }

    public GregorianCalendar(int year, int month, int dayOfMonth, int hourOfDay, int minute) {
    }

    public GregorianCalendar(int year, int month, int dayOfMonth, int hourOfDay, int minute, int second) {
    }

    public void setGregorianChange(@GuardSatisfied GregorianCalendar this, Date date);

    public final Date getGregorianChange();

    @Pure
    public boolean isLeapYear(@GuardSatisfied GregorianCalendar this, int year);

    @Override
    public String getCalendarType();

    @Pure
    @Override
    public boolean equals(@GuardSatisfied GregorianCalendar this, @GuardSatisfied @Nullable Object obj);

    @Pure
    @Override
    public int hashCode(@GuardSatisfied GregorianCalendar this);

    @Override
    public void add(@GuardSatisfied GregorianCalendar this, int field, int amount);

    @Override
    public void roll(@GuardSatisfied GregorianCalendar this, int field, boolean up);

    @Override
    public void roll(@GuardSatisfied GregorianCalendar this, int field, int amount);

    @Override
    public int getMinimum(int field);

    @Override
    public int getMaximum(int field);

    @Override
    public int getGreatestMinimum(int field);

    @Override
    public int getLeastMaximum(int field);

    @Override
    public int getActualMinimum(int field);

    @Override
    public int getActualMaximum(int field);

    @SideEffectFree
    @Override
    public Object clone(@GuardSatisfied GregorianCalendar this);

    @Override
    public TimeZone getTimeZone();

    @Override
    public void setTimeZone(@GuardSatisfied GregorianCalendar this, TimeZone zone);

    @Override
    public final boolean isWeekDateSupported();

    @Override
    public int getWeekYear();

    @Override
    public void setWeekDate(int weekYear, int weekOfYear, int dayOfWeek);

    @Override
    public int getWeeksInWeekYear();

    @Override
    protected void computeFields();

    @Override
    protected void computeTime();

    public ZonedDateTime toZonedDateTime();

    public static GregorianCalendar from(ZonedDateTime zdt);
}
