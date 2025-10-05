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
package java.text;

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
import java.io.InvalidObjectException;
    @Positive
import java.text.spi.DateFormatProvider;
    @Positive
import java.util.Calendar;
    @Positive
import java.util.Date;
    @Positive
import java.util.GregorianCalendar;
    @Positive
import java.util.HashMap;
    @Positive
import java.util.Locale;
    @Positive
import java.util.Map;
    @Positive
import java.util.MissingResourceException;
    @Positive
import java.util.ResourceBundle;
    @Positive
import java.util.TimeZone;
    @Positive
import java.util.spi.LocaleServiceProvider;
    @Positive
import sun.util.locale.provider.LocaleProviderAdapter;
    @Positive
import sun.util.locale.provider.LocaleServiceProviderPool;

    @Positive
public abstract class DateFormat extends Format {

    @Positive
    protected Calendar calendar;

    @Positive
    protected NumberFormat numberFormat;

    @Positive
    public static final int ERA_FIELD;

    @Positive
    public static final int YEAR_FIELD;

    @Positive
    public static final int MONTH_FIELD;

    @Positive
    public static final int DATE_FIELD;

    @Positive
    public static final int HOUR_OF_DAY1_FIELD;

    @Positive
    public static final int HOUR_OF_DAY0_FIELD;

    @Positive
    public static final int MINUTE_FIELD;

    @Positive
    public static final int SECOND_FIELD;

    @Positive
    public static final int MILLISECOND_FIELD;

    @Positive
    public static final int DAY_OF_WEEK_FIELD;

    @Positive
    public static final int DAY_OF_YEAR_FIELD;

    @Positive
    public static final int DAY_OF_WEEK_IN_MONTH_FIELD;

    @Positive
    public static final int WEEK_OF_YEAR_FIELD;

    @Positive
    public static final int WEEK_OF_MONTH_FIELD;

    @Positive
    public static final int AM_PM_FIELD;

    @Positive
    public static final int HOUR1_FIELD;

    @Positive
    public static final int HOUR0_FIELD;

    @Positive
    public static final int TIMEZONE_FIELD;

    @Positive
    public final StringBuffer format(Object obj, StringBuffer toAppendTo, FieldPosition fieldPosition);

    @Positive
    public abstract StringBuffer format(Date date, StringBuffer toAppendTo, FieldPosition fieldPosition);

    @Positive
    public final String format(Date date);

    @Positive
    public Date parse(String source) throws ParseException;

    @Positive
    public abstract Date parse(String source, ParsePosition pos);

    @Positive
    public Object parseObject(String source, ParsePosition pos);

    @Positive
    public static final int FULL;

    @Positive
    public static final int LONG;

    @Positive
    public static final int MEDIUM;

    @Positive
    public static final int SHORT;

    @Positive
    public static final int DEFAULT;

    @Positive
    public static final DateFormat getTimeInstance();

    @Positive
    public static final DateFormat getTimeInstance(int style);

    @Positive
    public static final DateFormat getTimeInstance(int style, Locale aLocale);

    @Positive
    public static final DateFormat getDateInstance();

    @Positive
    public static final DateFormat getDateInstance(int style);

    @Positive
    public static final DateFormat getDateInstance(int style, Locale aLocale);

    @Positive
    public static final DateFormat getDateTimeInstance();

    @Positive
    public static final DateFormat getDateTimeInstance(int dateStyle, int timeStyle);

    @Positive
    public static final DateFormat getDateTimeInstance(int dateStyle, int timeStyle, Locale aLocale);

    @Positive
    public static final DateFormat getInstance();

    @Positive
    public static Locale[] getAvailableLocales();

    @Positive
    public void setCalendar(Calendar newCalendar);

    @Positive
    public Calendar getCalendar();

    @Positive
    public void setNumberFormat(NumberFormat newNumberFormat);

    @Positive
    public NumberFormat getNumberFormat();

    @Positive
    public void setTimeZone(TimeZone zone);

    @Positive
    public TimeZone getTimeZone();

    @Positive
    public void setLenient(boolean lenient);

    @Positive
    public boolean isLenient();

    @Positive
    public int hashCode();

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    public Object clone();

    @Positive
    protected DateFormat() {
    @Positive
    }

    @Positive
    public static class Field extends Format.Field {

    @Positive
        public static Field ofCalendarField(int calendarField);

    @Positive
        protected Field(String name, int calendarField) {
    @Positive
        }

    @Positive
        public int getCalendarField();

    @Positive
        @Override
    @Positive
        @java.io.Serial
    @Positive
        protected Object readResolve() throws InvalidObjectException;

    @Positive
        public static final Field ERA;

    @Positive
        public static final Field YEAR;

    @Positive
        public static final Field MONTH;

    @Positive
        public static final Field DAY_OF_MONTH;

    @Positive
        public static final Field HOUR_OF_DAY1;

    @Positive
        public static final Field HOUR_OF_DAY0;

    @Positive
        public static final Field MINUTE;

    @Positive
        public static final Field SECOND;

    @Positive
        public static final Field MILLISECOND;

    @Positive
        public static final Field DAY_OF_WEEK;

    @Positive
        public static final Field DAY_OF_YEAR;

    @Positive
        public static final Field DAY_OF_WEEK_IN_MONTH;

    @Positive
        public static final Field WEEK_OF_YEAR;

    @Positive
        public static final Field WEEK_OF_MONTH;

    @Positive
        public static final Field AM_PM;

    @Positive
        public static final Field HOUR1;

    @Positive
        public static final Field HOUR0;

    @Positive
        public static final Field TIME_ZONE;
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 0
