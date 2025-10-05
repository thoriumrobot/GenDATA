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
package java.time;

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
import static java.time.LocalTime.SECONDS_PER_DAY;
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
import static java.time.temporal.ChronoField.DAY_OF_YEAR;
    @Positive
import static java.time.temporal.ChronoField.EPOCH_DAY;
    @Positive
import static java.time.temporal.ChronoField.ERA;
    @Positive
import static java.time.temporal.ChronoField.MONTH_OF_YEAR;
    @Positive
import static java.time.temporal.ChronoField.PROLEPTIC_MONTH;
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
import java.time.chrono.ChronoLocalDate;
    @Positive
import java.time.chrono.IsoEra;
    @Positive
import java.time.chrono.IsoChronology;
    @Positive
import java.time.format.DateTimeFormatter;
    @Positive
import java.time.format.DateTimeParseException;
    @Positive
import java.time.temporal.ChronoField;
    @Positive
import java.time.temporal.ChronoUnit;
    @Positive
import java.time.temporal.Temporal;
    @Positive
import java.time.temporal.TemporalAccessor;
    @Positive
import java.time.temporal.TemporalAdjuster;
    @Positive
import java.time.temporal.TemporalAmount;
    @Positive
import java.time.temporal.TemporalField;
    @Positive
import java.time.temporal.TemporalQueries;
    @Positive
import java.time.temporal.TemporalQuery;
    @Positive
import java.time.temporal.TemporalUnit;
    @Positive
import java.time.temporal.UnsupportedTemporalTypeException;
    @Positive
import java.time.temporal.ValueRange;
    @Positive
import java.time.zone.ZoneOffsetTransition;
    @Positive
import java.time.zone.ZoneRules;
    @Positive
import java.util.Objects;
    @Positive
import java.util.stream.LongStream;
    @Positive
import java.util.stream.Stream;

    @Positive
@jdk.internal.ValueBased
    @Positive
public final class LocalDate implements Temporal, TemporalAdjuster, ChronoLocalDate, Serializable {

    @Positive
    public static final LocalDate MIN;

    @Positive
    public static final LocalDate MAX;

    @Positive
    public static final LocalDate EPOCH;

    @Positive
    public static LocalDate now();

    @Positive
    public static LocalDate now(ZoneId zone);

    @Positive
    public static LocalDate now(Clock clock);

    @Positive
    public static LocalDate of(int year, Month month, int dayOfMonth);

    @Positive
    public static LocalDate of(int year, int month, int dayOfMonth);

    @Positive
    public static LocalDate ofYearDay(int year, int dayOfYear);

    @Positive
    public static LocalDate ofInstant(Instant instant, ZoneId zone);

    @Positive
    public static LocalDate ofEpochDay(long epochDay);

    @Positive
    public static LocalDate from(TemporalAccessor temporal);

    @Positive
    public static LocalDate parse(CharSequence text);

    @Positive
    public static LocalDate parse(CharSequence text, DateTimeFormatter formatter);

    @Positive
    @Override
    @Positive
    public boolean isSupported(TemporalField field);

    @Positive
    @Override
    @Positive
    public boolean isSupported(TemporalUnit unit);

    @Positive
    @Override
    @Positive
    public ValueRange range(TemporalField field);

    @Positive
    @Override
    @Positive
    public int get(TemporalField field);

    @Positive
    @Override
    @Positive
    public long getLong(TemporalField field);

    @Positive
    @Override
    @Positive
    public IsoChronology getChronology();

    @Positive
    @Override
    @Positive
    public IsoEra getEra();

    @Positive
    public int getYear();

    @Positive
    public int getMonthValue();

    @Positive
    public Month getMonth();

    @Positive
    public int getDayOfMonth();

    @Positive
    public int getDayOfYear();

    @Positive
    public DayOfWeek getDayOfWeek();

    @Positive
    @Override
    @Positive
    public boolean isLeapYear();

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
    public LocalDate with(TemporalAdjuster adjuster);

    @Positive
    @Override
    @Positive
    public LocalDate with(TemporalField field, long newValue);

    @Positive
    public LocalDate withYear(int year);

    @Positive
    public LocalDate withMonth(int month);

    @Positive
    public LocalDate withDayOfMonth(int dayOfMonth);

    @Positive
    public LocalDate withDayOfYear(int dayOfYear);

    @Positive
    @Override
    @Positive
    public LocalDate plus(TemporalAmount amountToAdd);

    @Positive
    @Override
    @Positive
    public LocalDate plus(long amountToAdd, TemporalUnit unit);

    @Positive
    public LocalDate plusYears(long yearsToAdd);

    @Positive
    public LocalDate plusMonths(long monthsToAdd);

    @Positive
    public LocalDate plusWeeks(long weeksToAdd);

    @Positive
    public LocalDate plusDays(long daysToAdd);

    @Positive
    @Override
    @Positive
    public LocalDate minus(TemporalAmount amountToSubtract);

    @Positive
    @Override
    @Positive
    public LocalDate minus(long amountToSubtract, TemporalUnit unit);

    @Positive
    public LocalDate minusYears(long yearsToSubtract);

    @Positive
    public LocalDate minusMonths(long monthsToSubtract);

    @Positive
    public LocalDate minusWeeks(long weeksToSubtract);

    @Positive
    public LocalDate minusDays(long daysToSubtract);

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    @Override
    @Positive
    public <R> R query(TemporalQuery<R> query);

    @Positive
    @Override
    @Positive
    public Temporal adjustInto(Temporal temporal);

    @Positive
    @Override
    @Positive
    public long until(Temporal endExclusive, TemporalUnit unit);

    @Positive
    long daysUntil(LocalDate end);

    @Positive
    @Override
    @Positive
    public Period until(ChronoLocalDate endDateExclusive);

    @Positive
    public Stream<LocalDate> datesUntil(LocalDate endExclusive);

    @Positive
    public Stream<LocalDate> datesUntil(LocalDate endExclusive, Period step);

    @Positive
    @Override
    @Positive
    public String format(DateTimeFormatter formatter);

    @Positive
    @Override
    @Positive
    public LocalDateTime atTime(LocalTime time);

    @Positive
    public LocalDateTime atTime(int hour, int minute);

    @Positive
    public LocalDateTime atTime(int hour, int minute, int second);

    @Positive
    public LocalDateTime atTime(int hour, int minute, int second, int nanoOfSecond);

    @Positive
    public OffsetDateTime atTime(OffsetTime time);

    @Positive
    public LocalDateTime atStartOfDay();

    @Positive
    public ZonedDateTime atStartOfDay(ZoneId zone);

    @Positive
    @Override
    @Positive
    public long toEpochDay();

    @Positive
    public long toEpochSecond(LocalTime time, ZoneOffset offset);

    @Positive
    @Override
    @Positive
    public int compareTo(ChronoLocalDate other);

    @Positive
    int compareTo0(LocalDate otherDate);

    @Positive
    @Override
    @Positive
    public boolean isAfter(ChronoLocalDate other);

    @Positive
    @Override
    @Positive
    public boolean isBefore(ChronoLocalDate other);

    @Positive
    @Override
    @Positive
    public boolean isEqual(ChronoLocalDate other);

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
    @Override
    @Positive
    public String toString();

    @Positive
    void writeExternal(DataOutput out) throws IOException;

    @Positive
    static LocalDate readExternal(DataInput in) throws IOException;
    @Positive
}

// CFWR semantic augmentation - variant 1
