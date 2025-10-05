/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1999, 2021, Oracle and/or its affiliates. All rights reserved.
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
package javax.management.monitor;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import static com.sun.jmx.defaults.JmxProperties.MONITOR_LOGGER;
    @Positive
import com.sun.jmx.mbeanserver.GetPropertyAction;
    @Positive
import com.sun.jmx.mbeanserver.Introspector;
    @Positive
import java.io.IOException;
    @Positive
import java.security.AccessControlContext;
    @Positive
import java.security.AccessController;
    @Positive
import java.security.PrivilegedAction;
    @Positive
import java.security.ProtectionDomain;
    @Positive
import java.util.List;
    @Positive
import java.util.Map;
    @Positive
import java.util.WeakHashMap;
    @Positive
import java.util.concurrent.CopyOnWriteArrayList;
    @Positive
import java.util.concurrent.Executors;
    @Positive
import java.util.concurrent.Future;
    @Positive
import java.util.concurrent.LinkedBlockingQueue;
    @Positive
import java.util.concurrent.ScheduledExecutorService;
    @Positive
import java.util.concurrent.ScheduledFuture;
    @Positive
import java.util.concurrent.ThreadFactory;
    @Positive
import java.util.concurrent.ThreadPoolExecutor;
    @Positive
import java.util.concurrent.TimeUnit;
    @Positive
import java.util.concurrent.atomic.AtomicInteger;
    @Positive
import java.util.concurrent.atomic.AtomicLong;
    @Positive
import java.lang.System.Logger.Level;
    @Positive
import javax.management.AttributeNotFoundException;
    @Positive
import javax.management.InstanceNotFoundException;
    @Positive
import javax.management.IntrospectionException;
    @Positive
import javax.management.MBeanAttributeInfo;
    @Positive
import javax.management.MBeanException;
    @Positive
import javax.management.MBeanInfo;
    @Positive
import javax.management.MBeanRegistration;
    @Positive
import javax.management.MBeanServer;
    @Positive
import javax.management.MBeanServerConnection;
    @Positive
import javax.management.NotificationBroadcasterSupport;
    @Positive
import javax.management.ObjectName;
    @Positive
import javax.management.ReflectionException;
    @Positive
import static javax.management.monitor.MonitorNotification.*;

    @Positive
public abstract class Monitor extends NotificationBroadcasterSupport implements MonitorMBean, MBeanRegistration {

    @Positive
    public Monitor() {
    @Positive
    }

    @Positive
    static class ObservedObject {

    @Positive
        public ObservedObject(ObjectName observedObject) {
    @Positive
        }

    @Positive
        public final ObjectName getObservedObject();

    @Positive
        public final synchronized int getAlreadyNotified();

    @Positive
        public final synchronized void setAlreadyNotified(int alreadyNotified);

    @Positive
        public final synchronized Object getDerivedGauge();

    @Positive
        public final synchronized void setDerivedGauge(Object derivedGauge);

    @Positive
        public final synchronized long getDerivedGaugeTimeStamp();

    @Positive
        public final synchronized void setDerivedGaugeTimeStamp(long derivedGaugeTimeStamp);
    @Positive
    }

    @Positive
    protected static final int capacityIncrement;

    @Positive
    protected int elementCount;

    @Positive
    @Deprecated
    @Positive
    protected int alreadyNotified;

    @Positive
    protected int[] alreadyNotifieds;

    @Positive
    protected MBeanServer server;

    @Positive
    protected static final int RESET_FLAGS_ALREADY_NOTIFIED;

    @Positive
    protected static final int OBSERVED_OBJECT_ERROR_NOTIFIED;

    @Positive
    protected static final int OBSERVED_ATTRIBUTE_ERROR_NOTIFIED;

    @Positive
    protected static final int OBSERVED_ATTRIBUTE_TYPE_ERROR_NOTIFIED;

    @Positive
    protected static final int RUNTIME_ERROR_NOTIFIED;

    @Positive
    @Deprecated
    @Positive
    protected String dbgTag;

    @Positive
    public ObjectName preRegister(MBeanServer server, ObjectName name) throws Exception;

    @Positive
    public void postRegister(Boolean registrationDone);

    @Positive
    public void preDeregister() throws Exception;

    @Positive
    public void postDeregister();

    @Positive
    public abstract void start();

    @Positive
    public abstract void stop();

    @Positive
    @Deprecated
    @Positive
    public synchronized ObjectName getObservedObject();

    @Positive
    @Deprecated
    @Positive
    public synchronized void setObservedObject(ObjectName object) throws IllegalArgumentException;

    @Positive
    public synchronized void addObservedObject(ObjectName object) throws IllegalArgumentException;

    @Positive
    public synchronized void removeObservedObject(ObjectName object);

    @Positive
    @Pure
    @Positive
    public synchronized boolean containsObservedObject(ObjectName object);

    @Positive
    public synchronized ObjectName[] getObservedObjects();

    @Positive
    public synchronized String getObservedAttribute();

    @Positive
    public void setObservedAttribute(String attribute) throws IllegalArgumentException;

    @Positive
    public synchronized long getGranularityPeriod();

    @Positive
    public synchronized void setGranularityPeriod(long period) throws IllegalArgumentException;

    @Positive
    public synchronized boolean isActive();

    @Positive
    @SuppressWarnings("removal")
    @Positive
    void doStart();

    @Positive
    void doStop();

    @Positive
    synchronized Object getDerivedGauge(ObjectName object);

    @Positive
    synchronized long getDerivedGaugeTimeStamp(ObjectName object);

    @Positive
    Object getAttribute(MBeanServerConnection mbsc, ObjectName object, String attribute) throws AttributeNotFoundException, InstanceNotFoundException, MBeanException, ReflectionException, IOException;

    @Positive
    Comparable<?> getComparableFromAttribute(ObjectName object, String attribute, Object value) throws AttributeNotFoundException;

    @Positive
    boolean isComparableTypeValid(ObjectName object, String attribute, Comparable<?> value);

    @Positive
    String buildErrorNotification(ObjectName object, String attribute, Comparable<?> value);

    @Positive
    void onErrorNotification(MonitorNotification notification);

    @Positive
    Comparable<?> getDerivedGaugeFromComparable(ObjectName object, String attribute, Comparable<?> value);

    @Positive
    MonitorNotification buildAlarmNotification(ObjectName object, String attribute, Comparable<?> value);

    @Positive
    boolean isThresholdTypeValid(ObjectName object, String attribute, Comparable<?> value);

    @Positive
    static Class<? extends Number> classForType(NumericalType type);

    @Positive
    static boolean isValidForType(Object value, Class<? extends Number> c);

    @Positive
    synchronized ObservedObject getObservedObject(ObjectName object);

    @Positive
    ObservedObject createObservedObject(ObjectName object);

    @Positive
    synchronized void createAlreadyNotified();

    @Positive
    synchronized void updateDeprecatedAlreadyNotified();

    @Positive
    synchronized void updateAlreadyNotified(ObservedObject o, int index);

    @Positive
    synchronized boolean isAlreadyNotified(ObservedObject o, int mask);

    @Positive
    synchronized void setAlreadyNotified(ObservedObject o, int index, int mask, int[] an);

    @Positive
    synchronized void resetAlreadyNotified(ObservedObject o, int index, int mask);

    @Positive
    synchronized void resetAllAlreadyNotified(ObservedObject o, int index, int[] an);

    @Positive
    synchronized int computeAlreadyNotifiedIndex(ObservedObject o, int index, int[] an);

    @Positive
    private class SchedulerTask implements Runnable {

    @Positive
        public SchedulerTask() {
    @Positive
        }

    @Positive
        public void setMonitorTask(MonitorTask task);

    @Positive
        public void run();
    @Positive
    }

    @Positive
    private class MonitorTask implements Runnable {

    @Positive
        public MonitorTask() {
    @Positive
        }

    @Positive
        public Future<?> submit();

    @Positive
        @SuppressWarnings("removal")
    @Positive
        public void run();
    @Positive
    }

    @Positive
    private static class DaemonThreadFactory implements ThreadFactory {

    @Positive
        public DaemonThreadFactory(String poolName) {
    @Positive
        }

    @Positive
        public DaemonThreadFactory(String poolName, ThreadGroup threadGroup) {
    @Positive
        }

    @Positive
        public ThreadGroup getThreadGroup();

    @Positive
        public Thread newThread(Runnable r);
    @Positive
    }
    @Positive
}
