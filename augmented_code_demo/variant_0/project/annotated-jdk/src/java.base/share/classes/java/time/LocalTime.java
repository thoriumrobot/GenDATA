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
import static java.time.temporal.ChronoField.HOUR_OF_DAY;
    @Positive
import static java.time.temporal.ChronoField.MICRO_OF_DAY;
    @Positive
import static java.time.temporal.ChronoField.MINUTE_OF_HOUR;
    @Positive
import static java.time.temporal.ChronoField.NANO_OF_DAY;
    @Positive
import static java.time.temporal.ChronoField.NANO_OF_SECOND;
    @Positive
import static java.time.temporal.ChronoField.SECOND_OF_DAY;
    @Positive
import static java.time.temporal.ChronoField.SECOND_OF_MINUTE;
    @Positive
import static java.time.temporal.ChronoUnit.NANOS;
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
import java.util.Objects;

    @Positive
@jdk.internal.ValueBased
    @Positive
public final class LocalTime implements Temporal, TemporalAdjuster, Comparable<LocalTime>, Serializable {

    @Positive
    public static final LocalTime MIN;

    @Positive
    public static final LocalTime MAX;

    @Positive
    public static final LocalTime MIDNIGHT;

    @Positive
    public static final LocalTime NOON;

    @Positive
    public static LocalTime now();

    @Positive
    public static LocalTime now(ZoneId zone);

    @Positive
    public static LocalTime now(Clock clock);

    @Positive
    public static LocalTime of(int hour, int minute);

    @Positive
    public static LocalTime of(int hour, int minute, int second);

    @Positive
    public static LocalTime of(int hour, int minute, int second, int nanoOfSecond);

    @Positive
    public static LocalTime ofInstant(Instant instant, ZoneId zone);

    @Positive
    public static LocalTime ofSecondOfDay(long secondOfDay);

    @Positive
    public static LocalTime ofNanoOfDay(long nanoOfDay);

    @Positive
    public static LocalTime from(TemporalAccessor temporal);

    @Positive
    public static LocalTime parse(CharSequence text);

    @Positive
    public static LocalTime parse(CharSequence text, DateTimeFormatter formatter);

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
    public int getHour();

    @Positive
    public int getMinute();

    @Positive
    public int getSecond();

    @Positive
    public int getNano();

    @Positive
    @Override
    @Positive
    public LocalTime with(TemporalAdjuster adjuster);

    @Positive
    @Override
    @Positive
    public LocalTime with(TemporalField field, long newValue);

    @Positive
    public LocalTime withHour(int hour);

    @Positive
    public LocalTime withMinute(int minute);

    @Positive
    public LocalTime withSecond(int second);

    @Positive
    public LocalTime withNano(int nanoOfSecond);

    @Positive
    public LocalTime truncatedTo(TemporalUnit unit);

    @Positive
    @Override
    @Positive
    public LocalTime plus(TemporalAmount amountToAdd);

    @Positive
    @Override
    @Positive
    public LocalTime plus(long amountToAdd, TemporalUnit unit);

    @Positive
    public LocalTime plusHours(long hoursToAdd);

    @Positive
    public LocalTime plusMinutes(long minutesToAdd);

    @Positive
    public LocalTime plusSeconds(long secondstoAdd);

    @Positive
    public LocalTime plusNanos(long nanosToAdd);

    @Positive
    @Override
    @Positive
    public LocalTime minus(TemporalAmount amountToSubtract);

    @Positive
    @Override
    @Positive
    public LocalTime minus(long amountToSubtract, TemporalUnit unit);

    @Positive
    public LocalTime minusHours(long hoursToSubtract);

    @Positive
    public LocalTime minusMinutes(long minutesToSubtract);

    @Positive
    public LocalTime minusSeconds(long secondsToSubtract);

    @Positive
    public LocalTime minusNanos(long nanosToSubtract);

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
    public LocalDateTime atDate(LocalDate date);

    @Positive
    public OffsetTime atOffset(ZoneOffset offset);

    @Positive
    public int toSecondOfDay();

    @Positive
    public long toNanoOfDay();

    @Positive
    public long toEpochSecond(LocalDate date, ZoneOffset offset);

    @Positive
    @Override
    @Positive
    public int compareTo(LocalTime other);

    @Positive
    public boolean isAfter(LocalTime other);

    @Positive
    public boolean isBefore(LocalTime other);

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
    static LocalTime readExternal(DataInput in) throws IOException;
    @Positive
}

// CFWR semantic augmentation - variant 0
