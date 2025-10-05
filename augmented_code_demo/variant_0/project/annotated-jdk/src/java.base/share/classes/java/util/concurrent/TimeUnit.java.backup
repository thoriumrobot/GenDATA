/*
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
package java.util.concurrent;

import org.checkerframework.framework.qual.AnnotatedFor;
import java.time.Duration;
import java.time.temporal.ChronoUnit;
import java.util.Objects;

@AnnotatedFor({ "lock" })
public enum TimeUnit {

    NANOSECONDS(TimeUnit.NANO_SCALE),
    MICROSECONDS(TimeUnit.MICRO_SCALE),
    MILLISECONDS(TimeUnit.MILLI_SCALE),
    SECONDS(TimeUnit.SECOND_SCALE),
    MINUTES(TimeUnit.MINUTE_SCALE),
    HOURS(TimeUnit.HOUR_SCALE),
    DAYS(TimeUnit.DAY_SCALE);

    public long convert(long sourceDuration, TimeUnit sourceUnit);

    public long convert(Duration duration);

    public long toNanos(long duration);

    public long toMicros(long duration);

    public long toMillis(long duration);

    public long toSeconds(long duration);

    public long toMinutes(long duration);

    public long toHours(long duration);

    public long toDays(long duration);

    public void timedWait(Object obj, long timeout) throws InterruptedException;

    public void timedJoin(Thread thread, long timeout) throws InterruptedException;

    public void sleep(long timeout) throws InterruptedException;

    public ChronoUnit toChronoUnit();

    public static TimeUnit of(ChronoUnit chronoUnit);
}
