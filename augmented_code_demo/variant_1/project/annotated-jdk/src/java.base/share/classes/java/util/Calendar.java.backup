/*
 * Copyright (c) 1996, 2021, Oracle and/or its affiliates. All rights reserved.
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

import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.checker.lock.qual.GuardSatisfied;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.common.value.qual.IntRange;
import org.checkerframework.common.value.qual.IntVal;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OptionalDataException;
import java.io.Serializable;
import java.security.AccessControlContext;
import java.security.AccessController;
import java.security.PermissionCollection;
import java.security.PrivilegedActionException;
import java.security.PrivilegedExceptionAction;
import java.security.ProtectionDomain;
import java.text.DateFormat;
import java.text.DateFormatSymbols;
import java.time.Instant;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import sun.util.BuddhistCalendar;
import sun.util.calendar.ZoneInfo;
import sun.util.locale.provider.CalendarDataUtility;
import sun.util.locale.provider.LocaleProviderAdapter;
import sun.util.locale.provider.TimeZoneNameUtility;
import sun.util.spi.CalendarProvider;

@AnnotatedFor({ "lock", "nullness", "index" })
public abstract class Calendar implements Serializable, Cloneable, Comparable<Calendar> {

    @IntVal({ 0 })
    public static final int ERA;

    @IntVal({ 1 })
    public static final int YEAR;

    @IntVal({ 2 })
    public static final int MONTH;

    @IntVal({ 3 })
    public static final int WEEK_OF_YEAR;

    @IntVal({ 4 })
    public static final int WEEK_OF_MONTH;

    @IntVal({ 5 })
    public static final int DATE;

    @IntVal({ 5 })
    public static final int DAY_OF_MONTH;

    @IntVal({ 6 })
    public static final int DAY_OF_YEAR;

    @IntVal({ 7 })
    public static final int DAY_OF_WEEK;

    @IntVal({ 8 })
    public static final int DAY_OF_WEEK_IN_MONTH;

    @IntVal({ 9 })
    public static final int AM_PM;

    @IntVal({ 10 })
    public static final int HOUR;

    @IntVal({ 11 })
    public static final int HOUR_OF_DAY;

    @IntVal({ 12 })
    public static final int MINUTE;

    @IntVal({ 13 })
    public static final int SECOND;

    @IntVal({ 14 })
    public static final int MILLISECOND;

    @IntVal({ 15 })
    public static final int ZONE_OFFSET;

    @IntVal({ 16 })
    public static final int DST_OFFSET;

    @IntVal({ 17 })
    public static final int FIELD_COUNT;

    @IntVal({ 1 })
    public static final int SUNDAY;

    @IntVal({ 2 })
    public static final int MONDAY;

    @IntVal({ 3 })
    public static final int TUESDAY;

    @IntVal({ 4 })
    public static final int WEDNESDAY;

    @IntVal({ 5 })
    public static final int THURSDAY;

    @IntVal({ 6 })
    public static final int FRIDAY;

    @IntVal({ 7 })
    public static final int SATURDAY;

    @IntVal({ 0 })
    public static final int JANUARY;

    @IntVal({ 1 })
    public static final int FEBRUARY;

    @IntVal({ 2 })
    public static final int MARCH;

    @IntVal({ 3 })
    public static final int APRIL;

    @IntVal({ 4 })
    public static final int MAY;

    @IntVal({ 5 })
    public static final int JUNE;

    @IntVal({ 6 })
    public static final int JULY;

    @IntVal({ 7 })
    public static final int AUGUST;

    @IntVal({ 8 })
    public static final int SEPTEMBER;

    @IntVal({ 9 })
    public static final int OCTOBER;

    @IntVal({ 10 })
    public static final int NOVEMBER;

    @IntVal({ 11 })
    public static final int DECEMBER;

    @IntVal({ 12 })
    public static final int UNDECIMBER;

    @IntVal({ 0 })
    public static final int AM;

    @IntVal({ 1 })
    public static final int PM;

    @IntVal({ 0 })
    public static final int ALL_STYLES;

    @IntVal({ 1 })
    public static final int SHORT;

    @IntVal({ 2 })
    public static final int LONG;

    public static final int NARROW_FORMAT;

    public static final int NARROW_STANDALONE;

    public static final int SHORT_FORMAT;

    public static final int LONG_FORMAT;

    public static final int SHORT_STANDALONE;

    public static final int LONG_STANDALONE;

    @SuppressWarnings("ProtectedField")
    protected int[] fields;

    @SuppressWarnings("ProtectedField")
    protected boolean[] isSet;

    @SuppressWarnings("ProtectedField")
    protected long time;

    @SuppressWarnings("ProtectedField")
    protected boolean isTimeSet;

    @SuppressWarnings("ProtectedField")
    protected boolean areFieldsSet;

    public static class Builder {

        public Builder() {
        }

        public Builder setInstant(long instant);

        public Builder setInstant(Date instant);

        public Builder set(int field, int value);

        public Builder setFields(int... fieldValuePairs);

        public Builder setDate(int year, int month, int dayOfMonth);

        public Builder setTimeOfDay(int hourOfDay, int minute, int second);

        public Builder setTimeOfDay(int hourOfDay, int minute, int second, int millis);

        public Builder setWeekDate(int weekYear, int weekOfYear, int dayOfWeek);

        public Builder setTimeZone(TimeZone zone);

        public Builder setLenient(boolean lenient);

        public Builder setCalendarType(String type);

        public Builder setLocale(Locale locale);

        public Builder setWeekDefinition(int firstDayOfWeek, int minimalDaysInFirstWeek);

        public Calendar build();
    }

    protected Calendar() {
    }

    protected Calendar(TimeZone zone, Locale aLocale) {
    }

    public static Calendar getInstance();

    public static Calendar getInstance(TimeZone zone);

    public static Calendar getInstance(Locale aLocale);

    public static Calendar getInstance(TimeZone zone, Locale aLocale);

    public static synchronized Locale[] getAvailableLocales();

    protected abstract void computeTime();

    protected abstract void computeFields();

    public final Date getTime(@GuardSatisfied Calendar this);

    public final void setTime(@GuardSatisfied Calendar this, Date date);

    public long getTimeInMillis(@GuardSatisfied Calendar this);

    public void setTimeInMillis(@GuardSatisfied Calendar this, long millis);

    @Pure
    @NonNegative
    public int get(@GuardSatisfied Calendar this, @NonNegative int field);

    protected final int internalGet(int field);

    final void internalSet(int field, int value);

    public void set(@GuardSatisfied Calendar this, @NonNegative int field, int value);

    public final void set(@GuardSatisfied Calendar this, @NonNegative int year, @NonNegative int month, @NonNegative int date);

    public final void set(@GuardSatisfied Calendar this, @NonNegative int year, @NonNegative int month, @NonNegative int date, @NonNegative int hourOfDay, @NonNegative int minute);

    public final void set(@GuardSatisfied Calendar this, @NonNegative int year, @NonNegative int month, @NonNegative int date, @NonNegative int hourOfDay, @NonNegative int minute, @NonNegative int second);

    public final void clear(@GuardSatisfied Calendar this);

    public final void clear(@GuardSatisfied Calendar this, @NonNegative int field);

    @Pure
    public final boolean isSet(@GuardSatisfied Calendar this, @NonNegative int field);

    @Nullable
    public String getDisplayName(@GuardSatisfied Calendar this, @NonNegative int field, int style, Locale locale);

    @Nullable
    public Map<String, Integer> getDisplayNames(@GuardSatisfied Calendar this, @NonNegative int field, int style, Locale locale);

    boolean checkDisplayNameParams(int field, int style, int minStyle, int maxStyle, Locale locale, int fieldMask);

    protected void complete();

    final boolean isExternallySet(int field);

    final int getSetStateFields();

    final void setFieldsComputed(int fieldMask);

    final void setFieldsNormalized(int fieldMask);

    final boolean isPartiallyNormalized();

    final boolean isFullyNormalized();

    final void setUnnormalized();

    static boolean isFieldSet(int fieldMask, int field);

    final int selectFields();

    int getBaseStyle(int style);

    public static Set<String> getAvailableCalendarTypes();

    private static class AvailableCalendarTypes {
    }

    public String getCalendarType();

    @Pure
    @SuppressWarnings("EqualsWhichDoesntCheckParameterClass")
    @Override
    public boolean equals(@GuardSatisfied Calendar this, @GuardSatisfied @Nullable Object obj);

    @Pure
    @Override
    public int hashCode(@GuardSatisfied Calendar this);

    public boolean before(Object when);

    public boolean after(Object when);

    @Pure
    @Override
    public int compareTo(@GuardSatisfied Calendar this, @GuardSatisfied Calendar anotherCalendar);

    public abstract void add(@GuardSatisfied Calendar this, int field, int amount);

    public abstract void roll(@GuardSatisfied Calendar this, int field, boolean up);

    public void roll(@GuardSatisfied Calendar this, @NonNegative int field, int amount);

    public void setTimeZone(@GuardSatisfied Calendar this, TimeZone value);

    public TimeZone getTimeZone(@GuardSatisfied Calendar this);

    TimeZone getZone();

    void setZoneShared(boolean shared);

    public void setLenient(@GuardSatisfied Calendar this, boolean lenient);

    @Pure
    public boolean isLenient(@GuardSatisfied Calendar this);

    public void setFirstDayOfWeek(@GuardSatisfied Calendar this, @IntRange(from = 1, to = 7) int value);

    @IntRange(from = 1, to = 7)
    public int getFirstDayOfWeek(@GuardSatisfied Calendar this);

    public void setMinimalDaysInFirstWeek(@GuardSatisfied Calendar this, @IntRange(from = 1, to = 7) int value);

    @IntRange(from = 1, to = 7)
    public int getMinimalDaysInFirstWeek(@GuardSatisfied Calendar this);

    public boolean isWeekDateSupported();

    public int getWeekYear();

    public void setWeekDate(int weekYear, int weekOfYear, int dayOfWeek);

    public int getWeeksInWeekYear();

    public abstract int getMinimum(@GuardSatisfied Calendar this, @NonNegative int field);

    public abstract int getMaximum(@GuardSatisfied Calendar this, @NonNegative int field);

    public abstract int getGreatestMinimum(@GuardSatisfied Calendar this, @NonNegative int field);

    public abstract int getLeastMaximum(@GuardSatisfied Calendar this, @NonNegative int field);

    @NonNegative
    public int getActualMinimum(@GuardSatisfied Calendar this, @NonNegative int field);

    @NonNegative
    public int getActualMaximum(@GuardSatisfied Calendar this, @NonNegative int field);

    @SideEffectFree
    @Override
    public Object clone(@GuardSatisfied Calendar this);

    static String getFieldName(int field);

    @SideEffectFree
    @Override
    public String toString(@GuardSatisfied Calendar this);

    @SuppressWarnings("removal")
    private static class CalendarAccessControlContext {
    }

    public final Instant toInstant();
}
