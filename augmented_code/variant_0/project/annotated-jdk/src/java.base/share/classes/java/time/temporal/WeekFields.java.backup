/*
 * Copyright (c) 2012, 2020, Oracle and/or its affiliates. All rights reserved.
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
package java.time.temporal;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import static java.time.temporal.ChronoField.DAY_OF_MONTH;
import static java.time.temporal.ChronoField.DAY_OF_WEEK;
import static java.time.temporal.ChronoField.DAY_OF_YEAR;
import static java.time.temporal.ChronoField.MONTH_OF_YEAR;
import static java.time.temporal.ChronoField.YEAR;
import static java.time.temporal.ChronoUnit.DAYS;
import static java.time.temporal.ChronoUnit.FOREVER;
import static java.time.temporal.ChronoUnit.MONTHS;
import static java.time.temporal.ChronoUnit.WEEKS;
import static java.time.temporal.ChronoUnit.YEARS;
import java.io.IOException;
import java.io.InvalidObjectException;
import java.io.ObjectInputStream;
import java.io.Serializable;
import java.time.DateTimeException;
import java.time.DayOfWeek;
import java.time.chrono.ChronoLocalDate;
import java.time.chrono.Chronology;
import java.time.format.ResolverStyle;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.ResourceBundle;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import sun.util.locale.provider.CalendarDataUtility;
import sun.util.locale.provider.LocaleProviderAdapter;
import sun.util.locale.provider.LocaleResources;

public final class WeekFields implements Serializable {

    public static final WeekFields ISO;

    public static final WeekFields SUNDAY_START;

    public static final TemporalUnit WEEK_BASED_YEARS;

    public static WeekFields of(Locale locale);

    public static WeekFields of(DayOfWeek firstDayOfWeek, int minimalDaysInFirstWeek);

    public DayOfWeek getFirstDayOfWeek();

    public int getMinimalDaysInFirstWeek();

    public TemporalField dayOfWeek();

    public TemporalField weekOfMonth();

    public TemporalField weekOfYear();

    public TemporalField weekOfWeekBasedYear();

    public TemporalField weekBasedYear();

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object object);

    @Override
    public int hashCode();

    @Override
    public String toString();

    static class ComputedDayOfField implements TemporalField {

        static ComputedDayOfField ofDayOfWeekField(WeekFields weekDef);

        static ComputedDayOfField ofWeekOfMonthField(WeekFields weekDef);

        static ComputedDayOfField ofWeekOfYearField(WeekFields weekDef);

        static ComputedDayOfField ofWeekOfWeekBasedYearField(WeekFields weekDef);

        static ComputedDayOfField ofWeekBasedYearField(WeekFields weekDef);

        @Override
        public long getFrom(TemporalAccessor temporal);

        @SuppressWarnings("unchecked")
        @Override
        public <R extends Temporal> R adjustInto(R temporal, long newValue);

        @Override
        public ChronoLocalDate resolve(Map<TemporalField, Long> fieldValues, TemporalAccessor partialTemporal, ResolverStyle resolverStyle);

        @Override
        public String getDisplayName(Locale locale);

        @Override
        public TemporalUnit getBaseUnit();

        @Override
        public TemporalUnit getRangeUnit();

        @Override
        public boolean isDateBased();

        @Override
        public boolean isTimeBased();

        @Override
        public ValueRange range();

        @Override
        public boolean isSupportedBy(TemporalAccessor temporal);

        @Override
        public ValueRange rangeRefinedBy(TemporalAccessor temporal);

        @Override
        public String toString();
    }
}
