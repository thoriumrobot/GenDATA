/*
    @Positive
 * Copyright (c) 2012, 2020, Oracle and/or its affiliates. All rights reserved.
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
import static java.time.temporal.ChronoField.EPOCH_DAY;
    @Positive
import java.io.IOException;
    @Positive
import java.io.InvalidObjectException;
    @Positive
import java.io.ObjectInput;
    @Positive
import java.io.ObjectInputStream;
    @Positive
import java.io.ObjectOutput;
    @Positive
import java.io.Serializable;
    @Positive
import java.time.LocalTime;
    @Positive
import java.time.ZoneId;
    @Positive
import java.time.temporal.ChronoField;
    @Positive
import java.time.temporal.ChronoUnit;
    @Positive
import java.time.temporal.Temporal;
    @Positive
import java.time.temporal.TemporalAdjuster;
    @Positive
import java.time.temporal.TemporalField;
    @Positive
import java.time.temporal.TemporalUnit;
    @Positive
import java.time.temporal.ValueRange;
    @Positive
import java.util.Objects;

    @Positive
final class ChronoLocalDateTimeImpl<D extends ChronoLocalDate> implements ChronoLocalDateTime<D>, Temporal, TemporalAdjuster, Serializable {

    @Positive
    static <R extends ChronoLocalDate> ChronoLocalDateTimeImpl<R> of(R date, LocalTime time);

    @Positive
    static <R extends ChronoLocalDate> ChronoLocalDateTimeImpl<R> ensureValid(Chronology chrono, Temporal temporal);

    @Positive
    @Override
    @Positive
    public D toLocalDate();

    @Positive
    @Override
    @Positive
    public LocalTime toLocalTime();

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
    public int get(TemporalField field);

    @Positive
    @Override
    @Positive
    public long getLong(TemporalField field);

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    @Override
    @Positive
    public ChronoLocalDateTimeImpl<D> with(TemporalAdjuster adjuster);

    @Positive
    @Override
    @Positive
    public ChronoLocalDateTimeImpl<D> with(TemporalField field, long newValue);

    @Positive
    @Override
    @Positive
    public ChronoLocalDateTimeImpl<D> plus(long amountToAdd, TemporalUnit unit);

    @Positive
    ChronoLocalDateTimeImpl<D> plusSeconds(long seconds);

    @Positive
    @Override
    @Positive
    public ChronoZonedDateTime<D> atZone(ZoneId zone);

    @Positive
    @Override
    @Positive
    public long until(Temporal endExclusive, TemporalUnit unit);

    @Positive
    void writeExternal(ObjectOutput out) throws IOException;

    @Positive
    static ChronoLocalDateTime<?> readExternal(ObjectInput in) throws IOException, ClassNotFoundException;

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
}

// CFWR semantic augmentation - variant 1
