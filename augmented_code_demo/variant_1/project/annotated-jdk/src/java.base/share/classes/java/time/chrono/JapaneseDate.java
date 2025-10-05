/*
    @Positive
 * Copyright (c) 2012, 2019, Oracle and/or its affiliates. All rights reserved.
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
package java.time.chrono;

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
import static java.time.temporal.ChronoField.ALIGNED_DAY_OF_WEEK_IN_MONTH;
    @Positive
import static java.time.temporal.ChronoField.ALIGNED_DAY_OF_WEEK_IN_YEAR;
    @Positive
import static java.time.temporal.ChronoField.ALIGNED_WEEK_OF_MONTH;
    @Positive
import static java.time.temporal.ChronoField.ALIGNED_WEEK_OF_YEAR;
    @Positive
import static java.time.temporal.ChronoField.DAY_OF_MONTH;
    @Positive
import static java.time.temporal.ChronoField.MONTH_OF_YEAR;
    @Positive
import static java.time.temporal.ChronoField.YEAR;
    @Positive
import java.io.DataInput;
    @Positive
import java.io.DataOutput;
    @Positive
import java.io.IOException;
    @Positive
import java.io.InvalidObjectException;
    @Positive
import java.io.ObjectInputStream;
    @Positive
import java.io.Serializable;
    @Positive
import java.time.Clock;
    @Positive
import java.time.DateTimeException;
    @Positive
import java.time.LocalDate;
    @Positive
import java.time.LocalTime;
    @Positive
import java.time.Period;
    @Positive
import java.time.ZoneId;
    @Positive
import java.time.temporal.ChronoField;
    @Positive
import java.time.temporal.TemporalAccessor;
    @Positive
import java.time.temporal.TemporalAdjuster;
    @Positive
import java.time.temporal.TemporalAmount;
    @Positive
import java.time.temporal.TemporalField;
    @Positive
import java.time.temporal.TemporalQuery;
    @Positive
import java.time.temporal.TemporalUnit;
    @Positive
import java.time.temporal.UnsupportedTemporalTypeException;
    @Positive
import java.time.temporal.ValueRange;
    @Positive
import java.util.Calendar;
    @Positive
import java.util.Objects;
    @Positive
import sun.util.calendar.CalendarDate;
    @Positive
import sun.util.calendar.LocalGregorianCalendar;

    @Positive
@jdk.internal.ValueBased
    @Positive
public final class JapaneseDate extends ChronoLocalDateImpl<JapaneseDate> implements ChronoLocalDate, Serializable {

    @Positive
    public static JapaneseDate now();

    @Positive
    public static JapaneseDate now(ZoneId zone);

    @Positive
    public static JapaneseDate now(Clock clock);

    @Positive
    public static JapaneseDate of(JapaneseEra era, int yearOfEra, int month, int dayOfMonth);

    @Positive
    public static JapaneseDate of(int prolepticYear, int month, int dayOfMonth);

    @Positive
    static JapaneseDate ofYearDay(JapaneseEra era, int yearOfEra, int dayOfYear);

    @Positive
    public static JapaneseDate from(TemporalAccessor temporal);

    @Positive
    @Override
    @Positive
    public JapaneseChronology getChronology();

    @Positive
    @Override
    @Positive
    public JapaneseEra getEra();

    @Positive
    @Override
    @Positive
    public int lengthOfMonth();

    @Positive
    @Override
    @Positive
    public int lengthOfYear();

    @Positive
    @Override
    @Positive
    public boolean isSupported(TemporalField field);

    @Positive
    @Override
    @Positive
    public ValueRange range(TemporalField field);

    @Positive
    @Override
    @Positive
    public long getLong(TemporalField field);

    @Positive
    @Override
    @Positive
    public JapaneseDate with(TemporalField field, long newValue);

    @Positive
    @Override
    @Positive
    public JapaneseDate with(TemporalAdjuster adjuster);

    @Positive
    @Override
    @Positive
    public JapaneseDate plus(TemporalAmount amount);

    @Positive
    @Override
    @Positive
    public JapaneseDate minus(TemporalAmount amount);

    @Positive
    @Override
    @Positive
    JapaneseDate plusYears(long years);

    @Positive
    @Override
    @Positive
    JapaneseDate plusMonths(long months);

    @Positive
    @Override
    @Positive
    JapaneseDate plusWeeks(long weeksToAdd);

    @Positive
    @Override
    @Positive
    JapaneseDate plusDays(long days);

    @Positive
    @Override
    @Positive
    public JapaneseDate plus(long amountToAdd, TemporalUnit unit);

    @Positive
    @Override
    @Positive
    public JapaneseDate minus(long amountToAdd, TemporalUnit unit);

    @Positive
    @Override
    @Positive
    JapaneseDate minusYears(long yearsToSubtract);

    @Positive
    @Override
    @Positive
    JapaneseDate minusMonths(long monthsToSubtract);

    @Positive
    @Override
    @Positive
    JapaneseDate minusWeeks(long weeksToSubtract);

    @Positive
    @Override
    @Positive
    JapaneseDate minusDays(long daysToSubtract);

    @Positive
    @Override
    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    public final ChronoLocalDateTime<JapaneseDate> atTime(LocalTime localTime);

    @Positive
    @Override
    @Positive
    public ChronoPeriod until(ChronoLocalDate endDate);

    @Positive
    @Override
    @Positive
    public long toEpochDay();

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    @Override
    @Positive
    public int hashCode();

    @Positive
    void writeExternal(DataOutput out) throws IOException;

    @Positive
    static JapaneseDate readExternal(DataInput in) throws IOException;
    @Positive
}

// CFWR semantic augmentation - variant 1
