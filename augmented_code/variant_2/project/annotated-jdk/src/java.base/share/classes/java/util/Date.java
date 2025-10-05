/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1994, 2019, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.text.DateFormat;
    @Positive
import java.time.LocalDate;
    @Positive
import java.io.IOException;
    @Positive
import java.io.ObjectOutputStream;
    @Positive
import java.io.ObjectInputStream;
    @Positive
import java.lang.ref.SoftReference;
    @Positive
import java.time.Instant;
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
import sun.util.calendar.ZoneInfo;

    @Positive
@AnnotatedFor({ "lock", "nullness", "index" })
    @Positive
public class Date implements java.io.Serializable, Cloneable, Comparable<Date> {

    @Positive
    public Date() {
    @Positive
    }

    @Positive
    public Date(long date) {
    @Positive
    }

    @Positive
    @Deprecated
    @Positive
    public Date(int year, int month, int date) {
    @Positive
    }

    @Positive
    @Deprecated
    @Positive
    public Date(int year, int month, int date, int hrs, int min) {
    @Positive
    }

    @Positive
    @Deprecated
    @Positive
    public Date(int year, int month, int date, int hrs, int min, int sec) {
    @Positive
    }

    @Positive
    @Deprecated
    @Positive
    public Date(String s) {
    @Positive
    }

    @Positive
    @SideEffectFree
    @Positive
    public Object clone(@GuardSatisfied Date this);

    @Positive
    @Deprecated
    @Positive
    public static long UTC(int year, int month, int date, int hrs, int min, int sec);

    @Positive
    @Deprecated
    @Positive
    public static long parse(String s);

    @Positive
    @Deprecated
    @Positive
    public int getYear(@GuardSatisfied Date this);

    @Positive
    @Deprecated
    @Positive
    public void setYear(@GuardSatisfied Date this, int year);

    @Positive
    @Deprecated
    @Positive
    public int getMonth(@GuardSatisfied Date this);

    @Positive
    @Deprecated
    @Positive
    public void setMonth(@GuardSatisfied Date this, int month);

    @Positive
    @Deprecated
    @Positive
    public int getDate(@GuardSatisfied Date this);

    @Positive
    @Deprecated
    @Positive
    public void setDate(@GuardSatisfied Date this, int date);

    @Positive
    @Deprecated
    @Positive
    public int getDay(@GuardSatisfied Date this);

    @Positive
    @Deprecated
    @Positive
    public int getHours(@GuardSatisfied Date this);

    @Positive
    @Deprecated
    @Positive
    public void setHours(@GuardSatisfied Date this, int hours);

    @Positive
    @Deprecated
    @Positive
    public int getMinutes(@GuardSatisfied Date this);

    @Positive
    @Deprecated
    @Positive
    public void setMinutes(@GuardSatisfied Date this, int minutes);

    @Positive
    @Deprecated
    @Positive
    public int getSeconds(@GuardSatisfied Date this);

    @Positive
    @Deprecated
    @Positive
    public void setSeconds(@GuardSatisfied Date this, int seconds);

    @Positive
    public long getTime(@GuardSatisfied Date this);

    @Positive
    public void setTime(@GuardSatisfied Date this, long time);

    @Positive
    public boolean before(@GuardSatisfied Date this, Date when);

    @Positive
    public boolean after(@GuardSatisfied Date this, Date when);

    @Positive
    @Pure
    @Positive
    public boolean equals(@GuardSatisfied Date this, @GuardSatisfied @Nullable Object obj);

    @Positive
    static final long getMillisOf(Date date);

    @Positive
    @Pure
    @Positive
    public int compareTo(@GuardSatisfied Date this, @GuardSatisfied Date anotherDate);

    @Positive
    @Pure
    @Positive
    public int hashCode(@GuardSatisfied Date this);

    @Positive
    @SideEffectFree
    @Positive
    public String toString(@GuardSatisfied Date this);

    @Positive
    @Deprecated
    @Positive
    public String toLocaleString();

    @Positive
    @Deprecated
    @Positive
    public String toGMTString();

    @Positive
    @Deprecated
    @Positive
    public int getTimezoneOffset(@GuardSatisfied Date this);

    @Positive
    public static Date from(Instant instant);

    @Positive
    public Instant toInstant();
    @Positive
}
