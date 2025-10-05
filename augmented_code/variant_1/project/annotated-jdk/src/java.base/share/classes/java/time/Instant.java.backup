/*
 * Copyright (c) 2012, 2021, Oracle and/or its affiliates. All rights reserved.
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
import static java.time.LocalTime.NANOS_PER_SECOND;
import static java.time.LocalTime.SECONDS_PER_DAY;
import static java.time.LocalTime.SECONDS_PER_HOUR;
import static java.time.LocalTime.SECONDS_PER_MINUTE;
import static java.time.temporal.ChronoField.INSTANT_SECONDS;
import static java.time.temporal.ChronoField.MICRO_OF_SECOND;
import static java.time.temporal.ChronoField.MILLI_OF_SECOND;
import static java.time.temporal.ChronoField.NANO_OF_SECOND;
import static java.time.temporal.ChronoUnit.DAYS;
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
public final class Instant implements Temporal, TemporalAdjuster, Comparable<Instant>, Serializable {

    public static final Instant EPOCH;

    public static final Instant MIN;

    public static final Instant MAX;

    public static Instant now();

    public static Instant now(Clock clock);

    public static Instant ofEpochSecond(long epochSecond);

    public static Instant ofEpochSecond(long epochSecond, long nanoAdjustment);

    public static Instant ofEpochMilli(long epochMilli);

    public static Instant from(TemporalAccessor temporal);

    public static Instant parse(final CharSequence text);

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

    public long getEpochSecond();

    public int getNano();

    @Override
    public Instant with(TemporalAdjuster adjuster);

    @Override
    public Instant with(TemporalField field, long newValue);

    public Instant truncatedTo(TemporalUnit unit);

    @Override
    public Instant plus(TemporalAmount amountToAdd);

    @Override
    public Instant plus(long amountToAdd, TemporalUnit unit);

    public Instant plusSeconds(long secondsToAdd);

    public Instant plusMillis(long millisToAdd);

    public Instant plusNanos(long nanosToAdd);

    @Override
    public Instant minus(TemporalAmount amountToSubtract);

    @Override
    public Instant minus(long amountToSubtract, TemporalUnit unit);

    public Instant minusSeconds(long secondsToSubtract);

    public Instant minusMillis(long millisToSubtract);

    public Instant minusNanos(long nanosToSubtract);

    @SuppressWarnings("unchecked")
    @Override
    public <R> R query(TemporalQuery<R> query);

    @Override
    public Temporal adjustInto(Temporal temporal);

    @Override
    public long until(Temporal endExclusive, TemporalUnit unit);

    public OffsetDateTime atOffset(ZoneOffset offset);

    public ZonedDateTime atZone(ZoneId zone);

    public long toEpochMilli();

    @Override
    public int compareTo(Instant otherInstant);

    public boolean isAfter(Instant otherInstant);

    public boolean isBefore(Instant otherInstant);

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object other);

    @Override
    public int hashCode();

    @Override
    public String toString();

    void writeExternal(DataOutput out) throws IOException;

    static Instant readExternal(DataInput in) throws IOException;
}
