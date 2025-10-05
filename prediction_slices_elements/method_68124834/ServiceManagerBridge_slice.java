// Source-based slice around line 32
// Method: <com.google.common.util.concurrent.ServiceManagerBridge: ImmutableMultimap servicesByState()>


/**
 * Superinterface of {@link ServiceManager} to introduce a bridge method for {@code
 * servicesByState()}, to ensure binary compatibility with older Guava versions that specified
 * {@code servicesByState()} to return {@code ImmutableMultimap}.
 */
@J2ktIncompatible
@GwtIncompatible
interface ServiceManagerBridge {
  ImmutableMultimap<State, Service> servicesByState();
}
