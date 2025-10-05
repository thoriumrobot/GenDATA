/*
 * Copyright (c) 2012, 2019, Oracle and/or its affiliates. All rights reserved.
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
package java.time;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import static java.time.temporal.ChronoField.ERA;
import static java.time.temporal.ChronoField.MONTH_OF_YEAR;
import static java.time.temporal.ChronoField.PROLEPTIC_MONTH;
import static java.time.temporal.ChronoField.YEAR;
import static java.time.temporal.ChronoField.YEAR_OF_ERA;
import static java.time.temporal.ChronoUnit.CENTURIES;
import static java.time.temporal.ChronoUnit.DECADES;
import static java.time.temporal.ChronoUnit.ERAS;
import static java.time.temporal.ChronoUnit.MILLENNIA;
import static java.time.temporal.ChronoUnit.MONTHS;
import static java.time.temporal.ChronoUnit.YEARS;
import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.io.InvalidObjectException;
import java.io.ObjectInputStream;
import java.io.Serializable;
import java.time.chrono.Chronology;
import java.time.chrono.IsoChronology;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeFormatterBuilder;
import java.time.format.DateTimeParseException;
import java.time.format.SignStyle;
import java.time.temporal.ChronoField;
import java.time.temporal.ChronoUnit;
import java.time.temporal.Temporal;
import java.time.temporal.TemporalAccessor;
import java.time.temporal.TemporalAdjuster;
import java.time.temporal.TemporalAmount;
import java.time.temporal.TemporalField;
import java.time.temporal.TemporalQueries;
import java.time.temporal.TemporalQuery;
import java.time.temporal.TemporalUnit;
import java.time.temporal.UnsupportedTemporalTypeException;
import java.time.temporal.ValueRange;
import java.util.Objects;

@jdk.internal.ValueBased
public final class YearMonth implements Temporal, TemporalAdjuster, Comparable<YearMonth>, Serializable {

    public static YearMonth now();

    public static YearMonth now(ZoneId zone);

    public static YearMonth now(Clock clock);

    public static YearMonth of(int year, Month month);

    public static YearMonth of(int year, int month);

    public static YearMonth from(TemporalAccessor temporal);

    public static YearMonth parse(CharSequence text);

    public static YearMonth parse(CharSequence text, DateTimeFormatter formatter);

    @Override
    public boolean isSupported(TemporalField field);

    @Override
    public boolean isSupported(TemporalUnit unit);

    @Override
    public ValueRange range(TemporalField field);

    @Override
    public int get(TemporalField field);

    @Override
    public long getLong(TemporalField field);

    public int getYear();

    public int getMonthValue();

    public Month getMonth();

    public boolean isLeapYear();

    public boolean isValidDay(int dayOfMonth);

    public int lengthOfMonth();

    public int lengthOfYear();

    @Override
    public YearMonth with(TemporalAdjuster adjuster);

    @Override
    public YearMonth with(TemporalField field, long newValue);

    public YearMonth withYear(int year);

    public YearMonth withMonth(int month);

    @Override
    public YearMonth plus(TemporalAmount amountToAdd);

    @Override
    public YearMonth plus(long amountToAdd, TemporalUnit unit);

    public YearMonth plusYears(long yearsToAdd);

    public YearMonth plusMonths(long monthsToAdd);

    @Override
    public YearMonth minus(TemporalAmount amountToSubtract);

    @Override
    public YearMonth minus(long amountToSubtract, TemporalUnit unit);

    public YearMonth minusYears(long yearsToSubtract);

    public YearMonth minusMonths(long monthsToSubtract);

    @SuppressWarnings("unchecked")
    @Override
    public <R> R query(TemporalQuery<R> query);

    @Override
    public Temporal adjustInto(Temporal temporal);

    @Override
    public long until(Temporal endExclusive, TemporalUnit unit);

    public String format(DateTimeFormatter formatter);

    public LocalDate atDay(int dayOfMonth);

    public LocalDate atEndOfMonth();

    @Override
    public int compareTo(YearMonth other);

    public boolean isAfter(YearMonth other);

    public boolean isBefore(YearMonth other);

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    @Override
    public int hashCode();

    @Override
    public String toString();

    void writeExternal(DataOutput out) throws IOException;

    static YearMonth readExternal(DataInput in) throws IOException;
}
