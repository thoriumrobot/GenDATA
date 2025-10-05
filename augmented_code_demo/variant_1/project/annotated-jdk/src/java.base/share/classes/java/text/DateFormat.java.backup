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
package java.text;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.io.InvalidObjectException;
import java.text.spi.DateFormatProvider;
import java.util.Calendar;
import java.util.Date;
import java.util.GregorianCalendar;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.MissingResourceException;
import java.util.ResourceBundle;
import java.util.TimeZone;
import java.util.spi.LocaleServiceProvider;
import sun.util.locale.provider.LocaleProviderAdapter;
import sun.util.locale.provider.LocaleServiceProviderPool;

public abstract class DateFormat extends Format {

    protected Calendar calendar;

    protected NumberFormat numberFormat;

    public static final int ERA_FIELD;

    public static final int YEAR_FIELD;

    public static final int MONTH_FIELD;

    public static final int DATE_FIELD;

    public static final int HOUR_OF_DAY1_FIELD;

    public static final int HOUR_OF_DAY0_FIELD;

    public static final int MINUTE_FIELD;

    public static final int SECOND_FIELD;

    public static final int MILLISECOND_FIELD;

    public static final int DAY_OF_WEEK_FIELD;

    public static final int DAY_OF_YEAR_FIELD;

    public static final int DAY_OF_WEEK_IN_MONTH_FIELD;

    public static final int WEEK_OF_YEAR_FIELD;

    public static final int WEEK_OF_MONTH_FIELD;

    public static final int AM_PM_FIELD;

    public static final int HOUR1_FIELD;

    public static final int HOUR0_FIELD;

    public static final int TIMEZONE_FIELD;

    public final StringBuffer format(Object obj, StringBuffer toAppendTo, FieldPosition fieldPosition);

    public abstract StringBuffer format(Date date, StringBuffer toAppendTo, FieldPosition fieldPosition);

    public final String format(Date date);

    public Date parse(String source) throws ParseException;

    public abstract Date parse(String source, ParsePosition pos);

    public Object parseObject(String source, ParsePosition pos);

    public static final int FULL;

    public static final int LONG;

    public static final int MEDIUM;

    public static final int SHORT;

    public static final int DEFAULT;

    public static final DateFormat getTimeInstance();

    public static final DateFormat getTimeInstance(int style);

    public static final DateFormat getTimeInstance(int style, Locale aLocale);

    public static final DateFormat getDateInstance();

    public static final DateFormat getDateInstance(int style);

    public static final DateFormat getDateInstance(int style, Locale aLocale);

    public static final DateFormat getDateTimeInstance();

    public static final DateFormat getDateTimeInstance(int dateStyle, int timeStyle);

    public static final DateFormat getDateTimeInstance(int dateStyle, int timeStyle, Locale aLocale);

    public static final DateFormat getInstance();

    public static Locale[] getAvailableLocales();

    public void setCalendar(Calendar newCalendar);

    public Calendar getCalendar();

    public void setNumberFormat(NumberFormat newNumberFormat);

    public NumberFormat getNumberFormat();

    public void setTimeZone(TimeZone zone);

    public TimeZone getTimeZone();

    public void setLenient(boolean lenient);

    public boolean isLenient();

    public int hashCode();

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    public Object clone();

    protected DateFormat() {
    }

    public static class Field extends Format.Field {

        public static Field ofCalendarField(int calendarField);

        protected Field(String name, int calendarField) {
        }

        public int getCalendarField();

        @Override
        @java.io.Serial
        protected Object readResolve() throws InvalidObjectException;

        public static final Field ERA;

        public static final Field YEAR;

        public static final Field MONTH;

        public static final Field DAY_OF_MONTH;

        public static final Field HOUR_OF_DAY1;

        public static final Field HOUR_OF_DAY0;

        public static final Field MINUTE;

        public static final Field SECOND;

        public static final Field MILLISECOND;

        public static final Field DAY_OF_WEEK;

        public static final Field DAY_OF_YEAR;

        public static final Field DAY_OF_WEEK_IN_MONTH;

        public static final Field WEEK_OF_YEAR;

        public static final Field WEEK_OF_MONTH;

        public static final Field AM_PM;

        public static final Field HOUR1;

        public static final Field HOUR0;

        public static final Field TIME_ZONE;
    }
}
