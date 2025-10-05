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
import static java.time.LocalTime.NANOS_PER_HOUR;
import static java.time.LocalTime.NANOS_PER_MINUTE;
import static java.time.LocalTime.NANOS_PER_SECOND;
import static java.time.LocalTime.SECONDS_PER_DAY;
import static java.time.temporal.ChronoField.NANO_OF_DAY;
import static java.time.temporal.ChronoField.OFFSET_SECONDS;
import static java.time.temporal.ChronoUnit.NANOS;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;
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
import java.time.zone.ZoneRules;
import java.util.Objects;

@jdk.internal.ValueBased
public final class OffsetTime implements Temporal, TemporalAdjuster, Comparable<OffsetTime>, Serializable {

    public static final OffsetTime MIN;

    public static final OffsetTime MAX;

    public static OffsetTime now();

    public static OffsetTime now(ZoneId zone);

    public static OffsetTime now(Clock clock);

    public static OffsetTime of(LocalTime time, ZoneOffset offset);

    public static OffsetTime of(int hour, int minute, int second, int nanoOfSecond, ZoneOffset offset);

    public static OffsetTime ofInstant(Instant instant, ZoneId zone);

    public static OffsetTime from(TemporalAccessor temporal);

    public static OffsetTime parse(CharSequence text);

    public static OffsetTime parse(CharSequence text, DateTimeFormatter formatter);

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

    public ZoneOffset getOffset();

    public OffsetTime withOffsetSameLocal(ZoneOffset offset);

    public OffsetTime withOffsetSameInstant(ZoneOffset offset);

    public LocalTime toLocalTime();

    public int getHour();

    public int getMinute();

    public int getSecond();

    public int getNano();

    @Override
    public OffsetTime with(TemporalAdjuster adjuster);

    @Override
    public OffsetTime with(TemporalField field, long newValue);

    public OffsetTime withHour(int hour);

    public OffsetTime withMinute(int minute);

    public OffsetTime withSecond(int second);

    public OffsetTime withNano(int nanoOfSecond);

    public OffsetTime truncatedTo(TemporalUnit unit);

    @Override
    public OffsetTime plus(TemporalAmount amountToAdd);

    @Override
    public OffsetTime plus(long amountToAdd, TemporalUnit unit);

    public OffsetTime plusHours(long hours);

    public OffsetTime plusMinutes(long minutes);

    public OffsetTime plusSeconds(long seconds);

    public OffsetTime plusNanos(long nanos);

    @Override
    public OffsetTime minus(TemporalAmount amountToSubtract);

    @Override
    public OffsetTime minus(long amountToSubtract, TemporalUnit unit);

    public OffsetTime minusHours(long hours);

    public OffsetTime minusMinutes(long minutes);

    public OffsetTime minusSeconds(long seconds);

    public OffsetTime minusNanos(long nanos);

    @SuppressWarnings("unchecked")
    @Override
    public <R> R query(TemporalQuery<R> query);

    @Override
    public Temporal adjustInto(Temporal temporal);

    @Override
    public long until(Temporal endExclusive, TemporalUnit unit);

    public String format(DateTimeFormatter formatter);

    public OffsetDateTime atDate(LocalDate date);

    public long toEpochSecond(LocalDate date);

    @Override
    public int compareTo(OffsetTime other);

    public boolean isAfter(OffsetTime other);

    public boolean isBefore(OffsetTime other);

    public boolean isEqual(OffsetTime other);

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    @Override
    public int hashCode();

    @Override
    public String toString();

    void writeExternal(ObjectOutput out) throws IOException;

    static OffsetTime readExternal(ObjectInput in) throws IOException, ClassNotFoundException;
}
