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
import static java.time.temporal.ChronoField.ERA;
    @Positive
import static java.time.temporal.ChronoField.MONTH_OF_YEAR;
    @Positive
import static java.time.temporal.ChronoField.PROLEPTIC_MONTH;
    @Positive
import static java.time.temporal.ChronoField.YEAR;
    @Positive
import static java.time.temporal.ChronoField.YEAR_OF_ERA;
    @Positive
import static java.time.temporal.ChronoUnit.CENTURIES;
    @Positive
import static java.time.temporal.ChronoUnit.DECADES;
    @Positive
import static java.time.temporal.ChronoUnit.ERAS;
    @Positive
import static java.time.temporal.ChronoUnit.MILLENNIA;
    @Positive
import static java.time.temporal.ChronoUnit.MONTHS;
    @Positive
import static java.time.temporal.ChronoUnit.YEARS;
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
import java.time.chrono.Chronology;
    @Positive
import java.time.chrono.IsoChronology;
    @Positive
import java.time.format.DateTimeFormatter;
    @Positive
import java.time.format.DateTimeFormatterBuilder;
    @Positive
import java.time.format.DateTimeParseException;
    @Positive
import java.time.format.SignStyle;
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
import java.util.Objects;

    @Positive
@jdk.internal.ValueBased
    @Positive
public final class YearMonth implements Temporal, TemporalAdjuster, Comparable<YearMonth>, Serializable {

    @Positive
    public static YearMonth now();

    @Positive
    public static YearMonth now(ZoneId zone);

    @Positive
    public static YearMonth now(Clock clock);

    @Positive
    public static YearMonth of(int year, Month month);

    @Positive
    public static YearMonth of(int year, int month);

    @Positive
    public static YearMonth from(TemporalAccessor temporal);

    @Positive
    public static YearMonth parse(CharSequence text);

    @Positive
    public static YearMonth parse(CharSequence text, DateTimeFormatter formatter);

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
    public int getYear();

    @Positive
    public int getMonthValue();

    @Positive
    public Month getMonth();

    @Positive
    public boolean isLeapYear();

    @Positive
    public boolean isValidDay(int dayOfMonth);

    @Positive
    public int lengthOfMonth();

    @Positive
    public int lengthOfYear();

    @Positive
    @Override
    @Positive
    public YearMonth with(TemporalAdjuster adjuster);

    @Positive
    @Override
    @Positive
    public YearMonth with(TemporalField field, long newValue);

    @Positive
    public YearMonth withYear(int year);

    @Positive
    public YearMonth withMonth(int month);

    @Positive
    @Override
    @Positive
    public YearMonth plus(TemporalAmount amountToAdd);

    @Positive
    @Override
    @Positive
    public YearMonth plus(long amountToAdd, TemporalUnit unit);

    @Positive
    public YearMonth plusYears(long yearsToAdd);

    @Positive
    public YearMonth plusMonths(long monthsToAdd);

    @Positive
    @Override
    @Positive
    public YearMonth minus(TemporalAmount amountToSubtract);

    @Positive
    @Override
    @Positive
    public YearMonth minus(long amountToSubtract, TemporalUnit unit);

    @Positive
    public YearMonth minusYears(long yearsToSubtract);

    @Positive
    public YearMonth minusMonths(long monthsToSubtract);

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
    public String format(DateTimeFormatter formatter);

    @Positive
    public LocalDate atDay(int dayOfMonth);

    @Positive
    public LocalDate atEndOfMonth();

    @Positive
    @Override
    @Positive
    public int compareTo(YearMonth other);

    @Positive
    public boolean isAfter(YearMonth other);

    @Positive
    public boolean isBefore(YearMonth other);

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
    static YearMonth readExternal(DataInput in) throws IOException;
    @Positive
}

// CFWR semantic augmentation - variant 0
