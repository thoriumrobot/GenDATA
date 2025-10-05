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
import static java.time.LocalTime.NANOS_PER_HOUR;
    @Positive
import static java.time.LocalTime.NANOS_PER_MINUTE;
    @Positive
import static java.time.LocalTime.NANOS_PER_SECOND;
    @Positive
import static java.time.LocalTime.SECONDS_PER_DAY;
    @Positive
import static java.time.temporal.ChronoField.NANO_OF_DAY;
    @Positive
import static java.time.temporal.ChronoField.OFFSET_SECONDS;
    @Positive
import static java.time.temporal.ChronoUnit.NANOS;
    @Positive
import java.io.IOException;
    @Positive
import java.io.ObjectInput;
    @Positive
import java.io.ObjectOutput;
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
import java.time.zone.ZoneRules;
    @Positive
import java.util.Objects;

    @Positive
@jdk.internal.ValueBased
    @Positive
public final class OffsetTime implements Temporal, TemporalAdjuster, Comparable<OffsetTime>, Serializable {

    @Positive
    public static final OffsetTime MIN;

    @Positive
    public static final OffsetTime MAX;

    @Positive
    public static OffsetTime now();

    @Positive
    public static OffsetTime now(ZoneId zone);

    @Positive
    public static OffsetTime now(Clock clock);

    @Positive
    public static OffsetTime of(LocalTime time, ZoneOffset offset);

    @Positive
    public static OffsetTime of(int hour, int minute, int second, int nanoOfSecond, ZoneOffset offset);

    @Positive
    public static OffsetTime ofInstant(Instant instant, ZoneId zone);

    @Positive
    public static OffsetTime from(TemporalAccessor temporal);

    @Positive
    public static OffsetTime parse(CharSequence text);

    @Positive
    public static OffsetTime parse(CharSequence text, DateTimeFormatter formatter);

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
    public ZoneOffset getOffset();

    @Positive
    public OffsetTime withOffsetSameLocal(ZoneOffset offset);

    @Positive
    public OffsetTime withOffsetSameInstant(ZoneOffset offset);

    @Positive
    public LocalTime toLocalTime();

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
    public OffsetTime with(TemporalAdjuster adjuster);

    @Positive
    @Override
    @Positive
    public OffsetTime with(TemporalField field, long newValue);

    @Positive
    public OffsetTime withHour(int hour);

    @Positive
    public OffsetTime withMinute(int minute);

    @Positive
    public OffsetTime withSecond(int second);

    @Positive
    public OffsetTime withNano(int nanoOfSecond);

    @Positive
    public OffsetTime truncatedTo(TemporalUnit unit);

    @Positive
    @Override
    @Positive
    public OffsetTime plus(TemporalAmount amountToAdd);

    @Positive
    @Override
    @Positive
    public OffsetTime plus(long amountToAdd, TemporalUnit unit);

    @Positive
    public OffsetTime plusHours(long hours);

    @Positive
    public OffsetTime plusMinutes(long minutes);

    @Positive
    public OffsetTime plusSeconds(long seconds);

    @Positive
    public OffsetTime plusNanos(long nanos);

    @Positive
    @Override
    @Positive
    public OffsetTime minus(TemporalAmount amountToSubtract);

    @Positive
    @Override
    @Positive
    public OffsetTime minus(long amountToSubtract, TemporalUnit unit);

    @Positive
    public OffsetTime minusHours(long hours);

    @Positive
    public OffsetTime minusMinutes(long minutes);

    @Positive
    public OffsetTime minusSeconds(long seconds);

    @Positive
    public OffsetTime minusNanos(long nanos);

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
    public OffsetDateTime atDate(LocalDate date);

    @Positive
    public long toEpochSecond(LocalDate date);

    @Positive
    @Override
    @Positive
    public int compareTo(OffsetTime other);

    @Positive
    public boolean isAfter(OffsetTime other);

    @Positive
    public boolean isBefore(OffsetTime other);

    @Positive
    public boolean isEqual(OffsetTime other);

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
    void writeExternal(ObjectOutput out) throws IOException;

    @Positive
    static OffsetTime readExternal(ObjectInput in) throws IOException, ClassNotFoundException;
    @Positive
}

// CFWR semantic augmentation - variant 1
