# frozen_string_literal: true

require 'etc'
require 'fileutils'
require 'rake/clean'

begin
  require 'colorize'
rescue LoadError
  warn 'Install the colorize gem: gem install colorize'
  exit 1
end

# Avoid removing files named "core" from included libraries such as Eigen.
CLEAN.clear_exclude.exclude { |fn| fn.pathmap('%f').downcase == 'core' }

# Optional shared configuration, searched in parent directories.
config_files = [
  File.expand_path('../Rakefile_configure.rb', __dir__),
  File.expand_path('../../Rakefile_configure.rb', __dir__)
]

if (config = config_files.find { |path| File.exist?(path) })
  require config
else
  COMPILE_DEBUG      = false unless defined?(COMPILE_DEBUG)
  COMPILE_DYNAMIC    = false unless defined?(COMPILE_DYNAMIC)
  COMPILE_EXECUTABLE = true  unless defined?(COMPILE_EXECUTABLE)
end

OS = case RUBY_PLATFORM
     when /darwin/       then :mac
     when /linux|cygwin/ then :linux
     when /msys/         then :mingw
     else                     :win
     end

BUILD_TYPE = COMPILE_DEBUG ? 'Debug' : 'Release'

BASE_BUILD_OPTIONS = [
  "-DCMAKE_BUILD_TYPE=#{BUILD_TYPE}",
  "-DUTILS_BUILD_SHARED=#{COMPILE_DYNAMIC ? 'ON' : 'OFF'}",
  "-DUTILS_UPDATE_3RDPARTY=OFF",
  "-DUTILS_ALLOW_NETWORK_FETCH=OFF"
].freeze

def cmake_options(enable_tests:)
  (BASE_BUILD_OPTIONS + [
    "-DUTILS_ENABLE_TESTS=#{enable_tests ? 'ON' : 'OFF'}"
  ]).join(' ')
end

PARALLEL = if OS == :win
             ''
           else
             "--parallel #{Etc.nprocessors}"
           end

def in_dir(path)
  FileUtils.mkdir_p(path)
  Dir.chdir(path) { yield }
end

def visual_studio_arch
  cl = `where cl.exe 2>NUL`.lines.first.to_s.strip

  case cl
  when /(x64|amd64)\\cl\.exe/i then 'x64'
  when /(bin|x86|amd32)\\cl\.exe/i then 'x86'
  else
    raise 'Cannot determine Visual Studio architecture. Run from a Visual Studio Developer Prompt.'
  end
end

def configure_and_build(bits: nil, enable_tests: false, target: 'install')
  in_dir('build') do
    bits_opt = bits ? "-DBITS=#{bits}" : ''
    target_opt = target ? "--target #{target}" : ''
    sh "cmake -G Ninja #{bits_opt} #{cmake_options(enable_tests: enable_tests)} .."
    sh "cmake --build . --config #{BUILD_TYPE} #{target_opt} #{PARALLEL}"
  end
end

def build_and_run_tests
  bits = OS == :win ? visual_studio_arch : nil
  configure_and_build(bits: bits, enable_tests: true, target: nil)
  Dir.chdir('build') { sh "ctest -C #{BUILD_TYPE} --output-on-failure" }
end

desc 'Default task: build'
task default: :build

desc 'Build with CMake/Ninja'
task :build do
  puts "Build (#{OS})".green

  bits = OS == :win ? visual_studio_arch : nil
  configure_and_build(bits: bits, enable_tests: false, target: 'install')
end

desc 'Build tests and run CTest'
task :test do
  build_and_run_tests
end

desc 'Build tests and run CTest'
task :run do
  build_and_run_tests
end

desc 'Clean build artifacts'
task :clean do
  FileUtils.rm_rf(%w[build lib])
end

desc 'Hard reset repository and submodules'
task :git_submodules do
  sh 'git reset --hard'
  sh 'git submodule sync --recursive'
  sh 'git submodule update --init --checkout --recursive'
  sh 'git submodule foreach --recursive git reset --hard'
  sh 'git submodule foreach --recursive git clean -d -x -f'
end

desc 'Hard clean repository'
task :git_clean do
  sh 'git reset --hard'
  sh 'git clean -d -x -f'
end

desc 'Generate compile_commands.json and run cppcheck'
task :cppcheck do
  FileUtils.rm_rf('build')
  in_dir('build') do
    sh 'cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..'
    sh 'cppcheck --project=compile_commands.json'
  end
end

desc 'Run CPack from build/'
task :cpack do
  Dir.chdir('build') do
    sh 'cpack -C CPackConfig.cmake'
    sh 'cpack -C CPackSourceConfig.cmake'
  end
end

desc 'get 3rd software'
task :get_3rd do
  FileUtils.rm_rf('build')
  in_dir('build') do
    sh 'cmake -DUTILS_UPDATE_3RDPARTY=ON -DUTILS_ALLOW_NETWORK_FETCH=ON ..'
  end
end

# Compatibility aliases.
task build_osx: :build
task build_linux: :build
task build_mingw: :build
task build_win: :build
task clean_osx: :clean
task clean_linux: :clean
task clean_mingw: :clean
task clean_win: :clean
