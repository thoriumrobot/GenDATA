/*
    @Positive
 * Copyright (c) 1996, 2021, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.lock.qual.GuardSatisfied;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.common.value.qual.IntRange;
    @Positive
import org.checkerframework.common.value.qual.IntVal;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.IOException;
    @Positive
import java.io.ObjectInputStream;
    @Positive
import java.io.ObjectOutputStream;
    @Positive
import java.io.OptionalDataException;
    @Positive
import java.io.Serializable;
    @Positive
import java.security.AccessControlContext;
    @Positive
import java.security.AccessController;
    @Positive
import java.security.PermissionCollection;
    @Positive
import java.security.PrivilegedActionException;
    @Positive
import java.security.PrivilegedExceptionAction;
    @Positive
import java.security.ProtectionDomain;
    @Positive
import java.text.DateFormat;
    @Positive
import java.text.DateFormatSymbols;
    @Positive
import java.time.Instant;
    @Positive
import java.util.concurrent.ConcurrentHashMap;
    @Positive
import java.util.concurrent.ConcurrentMap;
    @Positive
import sun.util.BuddhistCalendar;
    @Positive
import sun.util.calendar.ZoneInfo;
    @Positive
import sun.util.locale.provider.CalendarDataUtility;
    @Positive
import sun.util.locale.provider.LocaleProviderAdapter;
    @Positive
import sun.util.locale.provider.TimeZoneNameUtility;
    @Positive
import sun.util.spi.CalendarProvider;

    @Positive
@AnnotatedFor({ "lock", "nullness", "index" })
    @Positive
public abstract class Calendar implements Serializable, Cloneable, Comparable<Calendar> {

    @Positive
    @IntVal({ 0 })
    @Positive
    public static final int ERA;

    @Positive
    @IntVal({ 1 })
    @Positive
    public static final int YEAR;

    @Positive
    @IntVal({ 2 })
    @Positive
    public static final int MONTH;

    @Positive
    @IntVal({ 3 })
    @Positive
    public static final int WEEK_OF_YEAR;

    @Positive
    @IntVal({ 4 })
    @Positive
    public static final int WEEK_OF_MONTH;

    @Positive
    @IntVal({ 5 })
    @Positive
    public static final int DATE;

    @Positive
    @IntVal({ 5 })
    @Positive
    public static final int DAY_OF_MONTH;

    @Positive
    @IntVal({ 6 })
    @Positive
    public static final int DAY_OF_YEAR;

    @Positive
    @IntVal({ 7 })
    @Positive
    public static final int DAY_OF_WEEK;

    @Positive
    @IntVal({ 8 })
    @Positive
    public static final int DAY_OF_WEEK_IN_MONTH;

    @Positive
    @IntVal({ 9 })
    @Positive
    public static final int AM_PM;

    @Positive
    @IntVal({ 10 })
    @Positive
    public static final int HOUR;

    @Positive
    @IntVal({ 11 })
    @Positive
    public static final int HOUR_OF_DAY;

    @Positive
    @IntVal({ 12 })
    @Positive
    public static final int MINUTE;

    @Positive
    @IntVal({ 13 })
    @Positive
    public static final int SECOND;

    @Positive
    @IntVal({ 14 })
    @Positive
    public static final int MILLISECOND;

    @Positive
    @IntVal({ 15 })
    @Positive
    public static final int ZONE_OFFSET;

    @Positive
    @IntVal({ 16 })
    @Positive
    public static final int DST_OFFSET;

    @Positive
    @IntVal({ 17 })
    @Positive
    public static final int FIELD_COUNT;

    @Positive
    @IntVal({ 1 })
    @Positive
    public static final int SUNDAY;

    @Positive
    @IntVal({ 2 })
    @Positive
    public static final int MONDAY;

    @Positive
    @IntVal({ 3 })
    @Positive
    public static final int TUESDAY;

    @Positive
    @IntVal({ 4 })
    @Positive
    public static final int WEDNESDAY;

    @Positive
    @IntVal({ 5 })
    @Positive
    public static final int THURSDAY;

    @Positive
    @IntVal({ 6 })
    @Positive
    public static final int FRIDAY;

    @Positive
    @IntVal({ 7 })
    @Positive
    public static final int SATURDAY;

    @Positive
    @IntVal({ 0 })
    @Positive
    public static final int JANUARY;

    @Positive
    @IntVal({ 1 })
    @Positive
    public static final int FEBRUARY;

    @Positive
    @IntVal({ 2 })
    @Positive
    public static final int MARCH;

    @Positive
    @IntVal({ 3 })
    @Positive
    public static final int APRIL;

    @Positive
    @IntVal({ 4 })
    @Positive
    public static final int MAY;

    @Positive
    @IntVal({ 5 })
    @Positive
    public static final int JUNE;

    @Positive
    @IntVal({ 6 })
    @Positive
    public static final int JULY;

    @Positive
    @IntVal({ 7 })
    @Positive
    public static final int AUGUST;

    @Positive
    @IntVal({ 8 })
    @Positive
    public static final int SEPTEMBER;

    @Positive
    @IntVal({ 9 })
    @Positive
    public static final int OCTOBER;

    @Positive
    @IntVal({ 10 })
    @Positive
    public static final int NOVEMBER;

    @Positive
    @IntVal({ 11 })
    @Positive
    public static final int DECEMBER;

    @Positive
    @IntVal({ 12 })
    @Positive
    public static final int UNDECIMBER;

    @Positive
    @IntVal({ 0 })
    @Positive
    public static final int AM;

    @Positive
    @IntVal({ 1 })
    @Positive
    public static final int PM;

    @Positive
    @IntVal({ 0 })
    @Positive
    public static final int ALL_STYLES;

    @Positive
    @IntVal({ 1 })
    @Positive
    public static final int SHORT;

    @Positive
    @IntVal({ 2 })
    @Positive
    public static final int LONG;

    @Positive
    public static final int NARROW_FORMAT;

    @Positive
    public static final int NARROW_STANDALONE;

    @Positive
    public static final int SHORT_FORMAT;

    @Positive
    public static final int LONG_FORMAT;

    @Positive
    public static final int SHORT_STANDALONE;

    @Positive
    public static final int LONG_STANDALONE;

    @Positive
    @SuppressWarnings("ProtectedField")
    @Positive
    protected int[] fields;

    @Positive
    @SuppressWarnings("ProtectedField")
    @Positive
    protected boolean[] isSet;

    @Positive
    @SuppressWarnings("ProtectedField")
    @Positive
    protected long time;

    @Positive
    @SuppressWarnings("ProtectedField")
    @Positive
    protected boolean isTimeSet;

    @Positive
    @SuppressWarnings("ProtectedField")
    @Positive
    protected boolean areFieldsSet;

    @Positive
    public static class Builder {

    @Positive
        public Builder() {
    @Positive
        }

    @Positive
        public Builder setInstant(long instant);

    @Positive
        public Builder setInstant(Date instant);

    @Positive
        public Builder set(int field, int value);

    @Positive
        public Builder setFields(int... fieldValuePairs);

    @Positive
        public Builder setDate(int year, int month, int dayOfMonth);

    @Positive
        public Builder setTimeOfDay(int hourOfDay, int minute, int second);

    @Positive
        public Builder setTimeOfDay(int hourOfDay, int minute, int second, int millis);

    @Positive
        public Builder setWeekDate(int weekYear, int weekOfYear, int dayOfWeek);

    @Positive
        public Builder setTimeZone(TimeZone zone);

    @Positive
        public Builder setLenient(boolean lenient);

    @Positive
        public Builder setCalendarType(String type);

    @Positive
        public Builder setLocale(Locale locale);

    @Positive
        public Builder setWeekDefinition(int firstDayOfWeek, int minimalDaysInFirstWeek);

    @Positive
        public Calendar build();
    @Positive
    }

    @Positive
    protected Calendar() {
    @Positive
    }

    @Positive
    protected Calendar(TimeZone zone, Locale aLocale) {
    @Positive
    }

    @Positive
    public static Calendar getInstance();

    @Positive
    public static Calendar getInstance(TimeZone zone);

    @Positive
    public static Calendar getInstance(Locale aLocale);

    @Positive
    public static Calendar getInstance(TimeZone zone, Locale aLocale);

    @Positive
    public static synchronized Locale[] getAvailableLocales();

    @Positive
    protected abstract void computeTime();

    @Positive
    protected abstract void computeFields();

    @Positive
    public final Date getTime(@GuardSatisfied Calendar this);

    @Positive
    public final void setTime(@GuardSatisfied Calendar this, Date date);

    @Positive
    public long getTimeInMillis(@GuardSatisfied Calendar this);

    @Positive
    public void setTimeInMillis(@GuardSatisfied Calendar this, long millis);

    @Positive
    @Pure
    @Positive
    @NonNegative
    @Positive
    public int get(@GuardSatisfied Calendar this, @NonNegative int field);

    @Positive
    protected final int internalGet(int field);

    @Positive
    final void internalSet(int field, int value);

    @Positive
    public void set(@GuardSatisfied Calendar this, @NonNegative int field, int value);

    @Positive
    public final void set(@GuardSatisfied Calendar this, @NonNegative int year, @NonNegative int month, @NonNegative int date);

    @Positive
    public final void set(@GuardSatisfied Calendar this, @NonNegative int year, @NonNegative int month, @NonNegative int date, @NonNegative int hourOfDay, @NonNegative int minute);

    @Positive
    public final void set(@GuardSatisfied Calendar this, @NonNegative int year, @NonNegative int month, @NonNegative int date, @NonNegative int hourOfDay, @NonNegative int minute, @NonNegative int second);

    @Positive
    public final void clear(@GuardSatisfied Calendar this);

    @Positive
    public final void clear(@GuardSatisfied Calendar this, @NonNegative int field);

    @Positive
    @Pure
    @Positive
    public final boolean isSet(@GuardSatisfied Calendar this, @NonNegative int field);

    @Positive
    @Nullable
    @Positive
    public String getDisplayName(@GuardSatisfied Calendar this, @NonNegative int field, int style, Locale locale);

    @Positive
    @Nullable
    @Positive
    public Map<String, Integer> getDisplayNames(@GuardSatisfied Calendar this, @NonNegative int field, int style, Locale locale);

    @Positive
    boolean checkDisplayNameParams(int field, int style, int minStyle, int maxStyle, Locale locale, int fieldMask);

    @Positive
    protected void complete();

    @Positive
    final boolean isExternallySet(int field);

    @Positive
    final int getSetStateFields();

    @Positive
    final void setFieldsComputed(int fieldMask);

    @Positive
    final void setFieldsNormalized(int fieldMask);

    @Positive
    final boolean isPartiallyNormalized();

    @Positive
    final boolean isFullyNormalized();

    @Positive
    final void setUnnormalized();

    @Positive
    static boolean isFieldSet(int fieldMask, int field);

    @Positive
    final int selectFields();

    @Positive
    int getBaseStyle(int style);

    @Positive
    public static Set<String> getAvailableCalendarTypes();

    @Positive
    private static class AvailableCalendarTypes {
    @Positive
    }

    @Positive
    public String getCalendarType();

    @Positive
    @Pure
    @Positive
    @SuppressWarnings("EqualsWhichDoesntCheckParameterClass")
    @Positive
    @Override
    @Positive
    public boolean equals(@GuardSatisfied Calendar this, @GuardSatisfied @Nullable Object obj);

    @Positive
    @Pure
    @Positive
    @Override
    @Positive
    public int hashCode(@GuardSatisfied Calendar this);

    @Positive
    public boolean before(Object when);

    @Positive
    public boolean after(Object when);

    @Positive
    @Pure
    @Positive
    @Override
    @Positive
    public int compareTo(@GuardSatisfied Calendar this, @GuardSatisfied Calendar anotherCalendar);

    @Positive
    public abstract void add(@GuardSatisfied Calendar this, int field, int amount);

    @Positive
    public abstract void roll(@GuardSatisfied Calendar this, int field, boolean up);

    @Positive
    public void roll(@GuardSatisfied Calendar this, @NonNegative int field, int amount);

    @Positive
    public void setTimeZone(@GuardSatisfied Calendar this, TimeZone value);

    @Positive
    public TimeZone getTimeZone(@GuardSatisfied Calendar this);

    @Positive
    TimeZone getZone();

    @Positive
    void setZoneShared(boolean shared);

    @Positive
    public void setLenient(@GuardSatisfied Calendar this, boolean lenient);

    @Positive
    @Pure
    @Positive
    public boolean isLenient(@GuardSatisfied Calendar this);

    @Positive
    public void setFirstDayOfWeek(@GuardSatisfied Calendar this, @IntRange(from = 1, to = 7) int value);

    @Positive
    @IntRange(from = 1, to = 7)
    @Positive
    public int getFirstDayOfWeek(@GuardSatisfied Calendar this);

    @Positive
    public void setMinimalDaysInFirstWeek(@GuardSatisfied Calendar this, @IntRange(from = 1, to = 7) int value);

    @Positive
    @IntRange(from = 1, to = 7)
    @Positive
    public int getMinimalDaysInFirstWeek(@GuardSatisfied Calendar this);

    @Positive
    public boolean isWeekDateSupported();

    @Positive
    public int getWeekYear();

    @Positive
    public void setWeekDate(int weekYear, int weekOfYear, int dayOfWeek);

    @Positive
    public int getWeeksInWeekYear();

    @Positive
    public abstract int getMinimum(@GuardSatisfied Calendar this, @NonNegative int field);

    @Positive
    public abstract int getMaximum(@GuardSatisfied Calendar this, @NonNegative int field);

    @Positive
    public abstract int getGreatestMinimum(@GuardSatisfied Calendar this, @NonNegative int field);

    @Positive
    public abstract int getLeastMaximum(@GuardSatisfied Calendar this, @NonNegative int field);

    @Positive
    @NonNegative
    @Positive
    public int getActualMinimum(@GuardSatisfied Calendar this, @NonNegative int field);

    @Positive
    @NonNegative
    @Positive
    public int getActualMaximum(@GuardSatisfied Calendar this, @NonNegative int field);

    @Positive
    @SideEffectFree
    @Positive
    @Override
    @Positive
    public Object clone(@GuardSatisfied Calendar this);

    @Positive
    static String getFieldName(int field);

    @Positive
    @SideEffectFree
    @Positive
    @Override
    @Positive
    public String toString(@GuardSatisfied Calendar this);

    @Positive
    @SuppressWarnings("removal")
    @Positive
    private static class CalendarAccessControlContext {
    @Positive
    }

    @Positive
    public final Instant toInstant();
    @Positive
}

// CFWR semantic augmentation - variant 0
