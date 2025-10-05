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
import static java.time.temporal.ChronoField.HOUR_OF_DAY;
import static java.time.temporal.ChronoField.MICRO_OF_DAY;
import static java.time.temporal.ChronoField.MINUTE_OF_HOUR;
import static java.time.temporal.ChronoField.NANO_OF_DAY;
import static java.time.temporal.ChronoField.NANO_OF_SECOND;
import static java.time.temporal.ChronoField.SECOND_OF_DAY;
import static java.time.temporal.ChronoField.SECOND_OF_MINUTE;
import static java.time.temporal.ChronoUnit.NANOS;
import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.io.InvalidObjectException;
import java.io.ObjectInputStream;
import java.io.Serializable;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeParseException;
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
public final class LocalTime implements Temporal, TemporalAdjuster, Comparable<LocalTime>, Serializable {

    public static final LocalTime MIN;

    public static final LocalTime MAX;

    public static final LocalTime MIDNIGHT;

    public static final LocalTime NOON;

    public static LocalTime now();

    public static LocalTime now(ZoneId zone);

    public static LocalTime now(Clock clock);

    public static LocalTime of(int hour, int minute);

    public static LocalTime of(int hour, int minute, int second);

    public static LocalTime of(int hour, int minute, int second, int nanoOfSecond);

    public static LocalTime ofInstant(Instant instant, ZoneId zone);

    public static LocalTime ofSecondOfDay(long secondOfDay);

    public static LocalTime ofNanoOfDay(long nanoOfDay);

    public static LocalTime from(TemporalAccessor temporal);

    public static LocalTime parse(CharSequence text);

    public static LocalTime parse(CharSequence text, DateTimeFormatter formatter);

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

    public int getHour();

    public int getMinute();

    public int getSecond();

    public int getNano();

    @Override
    public LocalTime with(TemporalAdjuster adjuster);

    @Override
    public LocalTime with(TemporalField field, long newValue);

    public LocalTime withHour(int hour);

    public LocalTime withMinute(int minute);

    public LocalTime withSecond(int second);

    public LocalTime withNano(int nanoOfSecond);

    public LocalTime truncatedTo(TemporalUnit unit);

    @Override
    public LocalTime plus(TemporalAmount amountToAdd);

    @Override
    public LocalTime plus(long amountToAdd, TemporalUnit unit);

    public LocalTime plusHours(long hoursToAdd);

    public LocalTime plusMinutes(long minutesToAdd);

    public LocalTime plusSeconds(long secondstoAdd);

    public LocalTime plusNanos(long nanosToAdd);

    @Override
    public LocalTime minus(TemporalAmount amountToSubtract);

    @Override
    public LocalTime minus(long amountToSubtract, TemporalUnit unit);

    public LocalTime minusHours(long hoursToSubtract);

    public LocalTime minusMinutes(long minutesToSubtract);

    public LocalTime minusSeconds(long secondsToSubtract);

    public LocalTime minusNanos(long nanosToSubtract);

    @SuppressWarnings("unchecked")
    @Override
    public <R> R query(TemporalQuery<R> query);

    @Override
    public Temporal adjustInto(Temporal temporal);

    @Override
    public long until(Temporal endExclusive, TemporalUnit unit);

    public String format(DateTimeFormatter formatter);

    public LocalDateTime atDate(LocalDate date);

    public OffsetTime atOffset(ZoneOffset offset);

    public int toSecondOfDay();

    public long toNanoOfDay();

    public long toEpochSecond(LocalDate date, ZoneOffset offset);

    @Override
    public int compareTo(LocalTime other);

    public boolean isAfter(LocalTime other);

    public boolean isBefore(LocalTime other);

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    @Override
    public int hashCode();

    @Override
    public String toString();

    void writeExternal(DataOutput out) throws IOException;

    static LocalTime readExternal(DataInput in) throws IOException;
}
