/*
 * Copyright (c) 2012, 2020, Oracle and/or its affiliates. All rights reserved.
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
package java.time.chrono;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import static java.time.temporal.ChronoField.EPOCH_DAY;
import java.io.IOException;
import java.io.InvalidObjectException;
import java.io.ObjectInput;
import java.io.ObjectInputStream;
import java.io.ObjectOutput;
import java.io.Serializable;
import java.time.LocalTime;
import java.time.ZoneId;
import java.time.temporal.ChronoField;
import java.time.temporal.ChronoUnit;
import java.time.temporal.Temporal;
import java.time.temporal.TemporalAdjuster;
import java.time.temporal.TemporalField;
import java.time.temporal.TemporalUnit;
import java.time.temporal.ValueRange;
import java.util.Objects;

final class ChronoLocalDateTimeImpl<D extends ChronoLocalDate> implements ChronoLocalDateTime<D>, Temporal, TemporalAdjuster, Serializable {

    static <R extends ChronoLocalDate> ChronoLocalDateTimeImpl<R> of(R date, LocalTime time);

    static <R extends ChronoLocalDate> ChronoLocalDateTimeImpl<R> ensureValid(Chronology chrono, Temporal temporal);

    @Override
    public D toLocalDate();

    @Override
    public LocalTime toLocalTime();

    @Override
    public boolean isSupported(TemporalField field);

    @Override
    public ValueRange range(TemporalField field);

    @Override
    public int get(TemporalField field);

    @Override
    public long getLong(TemporalField field);

    @SuppressWarnings("unchecked")
    @Override
    public ChronoLocalDateTimeImpl<D> with(TemporalAdjuster adjuster);

    @Override
    public ChronoLocalDateTimeImpl<D> with(TemporalField field, long newValue);

    @Override
    public ChronoLocalDateTimeImpl<D> plus(long amountToAdd, TemporalUnit unit);

    ChronoLocalDateTimeImpl<D> plusSeconds(long seconds);

    @Override
    public ChronoZonedDateTime<D> atZone(ZoneId zone);

    @Override
    public long until(Temporal endExclusive, TemporalUnit unit);

    void writeExternal(ObjectOutput out) throws IOException;

    static ChronoLocalDateTime<?> readExternal(ObjectInput in) throws IOException, ClassNotFoundException;

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    @Override
    public int hashCode();

    @Override
    public String toString();
}
