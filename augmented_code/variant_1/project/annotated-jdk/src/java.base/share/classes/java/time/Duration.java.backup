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
import static java.time.LocalTime.MINUTES_PER_HOUR;
import static java.time.LocalTime.NANOS_PER_MILLI;
import static java.time.LocalTime.NANOS_PER_SECOND;
import static java.time.LocalTime.SECONDS_PER_DAY;
import static java.time.LocalTime.SECONDS_PER_HOUR;
import static java.time.LocalTime.SECONDS_PER_MINUTE;
import static java.time.temporal.ChronoField.NANO_OF_SECOND;
import static java.time.temporal.ChronoUnit.DAYS;
import static java.time.temporal.ChronoUnit.NANOS;
import static java.time.temporal.ChronoUnit.SECONDS;
import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.io.InvalidObjectException;
import java.io.ObjectInputStream;
import java.io.Serializable;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.math.RoundingMode;
import java.time.format.DateTimeParseException;
import java.time.temporal.ChronoField;
import java.time.temporal.ChronoUnit;
import java.time.temporal.Temporal;
import java.time.temporal.TemporalAmount;
import java.time.temporal.TemporalUnit;
import java.time.temporal.UnsupportedTemporalTypeException;
import java.util.List;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

@jdk.internal.ValueBased
public final class Duration implements TemporalAmount, Comparable<Duration>, Serializable {

    public static final Duration ZERO;

    private static class Lazy {
    }

    public static Duration ofDays(long days);

    public static Duration ofHours(long hours);

    public static Duration ofMinutes(long minutes);

    public static Duration ofSeconds(long seconds);

    public static Duration ofSeconds(long seconds, long nanoAdjustment);

    public static Duration ofMillis(long millis);

    public static Duration ofNanos(long nanos);

    public static Duration of(long amount, TemporalUnit unit);

    public static Duration from(TemporalAmount amount);

    public static Duration parse(CharSequence text);

    public static Duration between(Temporal startInclusive, Temporal endExclusive);

    @Override
    public long get(TemporalUnit unit);

    @Override
    public List<TemporalUnit> getUnits();

    private static class DurationUnits {
    }

    public boolean isZero();

    public boolean isNegative();

    public long getSeconds();

    public int getNano();

    public Duration withSeconds(long seconds);

    public Duration withNanos(int nanoOfSecond);

    public Duration plus(Duration duration);

    public Duration plus(long amountToAdd, TemporalUnit unit);

    public Duration plusDays(long daysToAdd);

    public Duration plusHours(long hoursToAdd);

    public Duration plusMinutes(long minutesToAdd);

    public Duration plusSeconds(long secondsToAdd);

    public Duration plusMillis(long millisToAdd);

    public Duration plusNanos(long nanosToAdd);

    public Duration minus(Duration duration);

    public Duration minus(long amountToSubtract, TemporalUnit unit);

    public Duration minusDays(long daysToSubtract);

    public Duration minusHours(long hoursToSubtract);

    public Duration minusMinutes(long minutesToSubtract);

    public Duration minusSeconds(long secondsToSubtract);

    public Duration minusMillis(long millisToSubtract);

    public Duration minusNanos(long nanosToSubtract);

    public Duration multipliedBy(long multiplicand);

    public Duration dividedBy(long divisor);

    public long dividedBy(Duration divisor);

    public Duration negated();

    public Duration abs();

    @Override
    public Temporal addTo(Temporal temporal);

    @Override
    public Temporal subtractFrom(Temporal temporal);

    public long toDays();

    public long toHours();

    public long toMinutes();

    public long toSeconds();

    public long toMillis();

    public long toNanos();

    public long toDaysPart();

    public int toHoursPart();

    public int toMinutesPart();

    public int toSecondsPart();

    public int toMillisPart();

    public int toNanosPart();

    public Duration truncatedTo(TemporalUnit unit);

    @Override
    public int compareTo(Duration otherDuration);

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object other);

    @Override
    public int hashCode();

    @Override
    public String toString();

    void writeExternal(DataOutput out) throws IOException;

    static Duration readExternal(DataInput in) throws IOException;
}
